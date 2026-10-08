"""Локальный Transformer-энкодер по плану docs/plan_transformer_encoder.md.

Контракт: positions, masks -> descriptors, centers, valid.
Две оси сравнения, каждая меняет ровно одно:
  A/B  — маска ПЕРВОГО attention-подслоя: свой сустав во времени (A) против
         всех суставов в том же временном окне (B). Параметры и максимальный
         временной контекст совпадают.
  R0/R1 — порядок сжатия в readout: суставы→время (R0) или время→суставы (R1).

Рецептивное поле: stem даёт ±1 кадр, каждый из L временных подслоёв ±W, окно
выдачи 10 кадров. При L=4, W=2 это 10 + 2*(1 + 4*2) = 28 кадров.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NJ = 23          # суставы после исключения нулевого root
STEM_R = 1       # радиус stem в кадрах


def _neighbours(h: torch.Tensor, w: int) -> torch.Tensor:
    """(B,T,J,d) -> (B,T,J,2w+1,d): для каждого кадра его временное окно.

    Реализовано разворачиванием, а не маской поверх полной матрицы всего
    ролика: локальность должна быть вычислительной, иначе экономии нет.
    """
    B, T, J, d = h.shape
    x = h.permute(0, 2, 3, 1).reshape(B * J * d, 1, T)
    x = F.pad(x, (w, w))
    x = x.unfold(2, 2 * w + 1, 1)                      # (B*J*d, 1, T, 2w+1)
    x = x.view(B, J, d, T, 2 * w + 1)
    return x.permute(0, 3, 1, 4, 2).contiguous()       # (B,T,J,2w+1,d)


class Attn(nn.Module):
    """Мультиголова с общими по суставам и кадрам Q/K/V.

    Работает на формах (M, Lq, d) × (M, Lk, d). Ключи НЕ разворачиваются под
    каждый запрос: запросы группируются так, чтобы ключи были общими внутри
    группы. Первая версия расширяла ключи через expand+reshape и съедала 32 ГБ
    на A100 — стоимость локальности должна быть вычислительной, а не
    материализованной.
    """

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in, kv_in, bias=None, key_mask=None):
        """q_in (M,Lq,d); kv_in (M,Lk,d); bias (heads,Lk); key_mask (M,Lk)."""
        M, Lq, d = q_in.shape
        Lk = kv_in.shape[1]
        q = self.q(q_in).view(M, Lq, self.h, self.dk).transpose(1, 2)
        k, v = self.kv(kv_in).view(M, Lk, 2, self.h, self.dk).permute(
            2, 0, 3, 1, 4).unbind(0)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)   # (M,h,Lq,Lk)
        if bias is not None:
            att = att + bias.view(1, self.h, 1, Lk)
        if key_mask is not None:
            km = key_mask.view(M, 1, 1, Lk)
            att = att.masked_fill(~km, float('-inf'))
            dead = ~key_mask.any(1).view(M, 1, 1, 1)
            att = att.masked_fill(dead, 0.0)
        w = self.drop(att.softmax(-1))
        if key_mask is not None:
            w = w * key_mask.view(M, 1, 1, Lk)
        out = (w @ v).transpose(1, 2).reshape(M, Lq, self.h * self.dk)
        return self.o(out)


class Block(nn.Module):
    """Временное внимание -> пространственное -> общий FFN, всё с residual."""

    def __init__(self, d, heads, window, ffn_mult, dropout, joint_first_layer):
        super().__init__()
        self.w = window
        self.joint_first = joint_first_layer      # True -> вариант B
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d) for _ in range(3))
        self.temporal = Attn(d, heads, dropout)
        self.spatial = Attn(d, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        # ЗНАКОВОЕ относительное смещение: у -2 и +2 свои параметры
        self.tbias = nn.Parameter(torch.zeros(heads, 2 * window + 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, tok_valid):
        B, T, J, d = h.shape
        W = 2 * self.w + 1
        x = self.n1(h)
        nb = _neighbours(x, self.w)                     # (B,T,J,W,d)
        nbv = _neighbours(tok_valid.unsqueeze(-1).float(), self.w
                          ).squeeze(-1) > 0.5           # (B,T,J,W)
        if self.joint_first:
            # B: запросы одного кадра делят общий набор ключей (все суставы x W)
            kv = nb.permute(0, 1, 3, 2, 4).reshape(B * T, W * J, d)
            km = nbv.permute(0, 1, 3, 2).reshape(B * T, W * J)
            bias = self.tbias.repeat_interleave(J, dim=1)
            q = x.reshape(B * T, J, d)
        else:
            # A: тот же сустав, окно по времени; ключей всего W на запрос
            kv = nb.reshape(B * T * J, W, d)
            km = nbv.reshape(B * T * J, W)
            bias = self.tbias
            q = x.reshape(B * T * J, 1, d)
        h = h + self.drop(self.temporal(q, kv, bias, km).view(B, T, J, d))

        x = self.n2(h).reshape(B * T, J, d)
        sm = tok_valid.reshape(B * T, J)
        h = h + self.drop(self.spatial(x, x, None, sm).view(B, T, J, d))
        return h + self.drop(self.ffn(self.n3(h)))


class HistoryMix(nn.Module):
    """Обмен между УЖЕ СОБРАННЫМИ историями суставов внутри одного окна вывода.

    Два режима с выровненной ёмкостью:
      attn  — R2: внимание между 23 историями, затем FFN;
      ffn   — контроль: тот же FFN плюс второй, поточечный, БЕЗ обмена между
              суставами. Ширина второго подобрана так, чтобы число параметров
              совпало с вариантом attn.
    Без такого контроля разность R2 − R1 нельзя приписать именно вниманию
    между историями: R2 добавляет и параметры, и обработку.
    """

    def __init__(self, d, heads, ffn_mult, dropout, mode="attn"):
        super().__init__()
        assert mode in ("attn", "ffn")
        self.mode = mode
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        if mode == "attn":
            self.mix = Attn(d, heads, dropout)
        else:
            # Attn несёт q (d^2+d), kv (2d^2+2d), o (d^2+d) = 4d^2 + 4d.
            # Поточечный FFN d -> h -> d несёт 2dh + h + d. Подбираем h так,
            # чтобы разница была минимальной.
            h = max(1, round((4 * d * d + 3 * d) / (2 * d + 1)))
            self.mix = nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(h, d))

    def forward(self, z, valid=None):           # (M, J, d), (M, J)
        x = self.n1(z)
        if self.mode == "attn":
            # Пустая история сустава не должна участвовать ключом: иначе
            # окно, где сустав не наблюдался, всё равно влияет на остальные.
            upd = self.mix(x, x, key_mask=valid)
        else:
            upd = self.mix(x)
        z = z + upd
        z = z + self.ffn(self.n2(z))
        if valid is not None:
            z = z * valid.unsqueeze(-1)
        return z


class LocalEncoder(nn.Module):
    def __init__(self, d=128, heads=4, layers=4, window=2, ffn_mult=4,
                 dropout=0.1, stride=10, variant="A", readout="R0",
                 pelvis_dim=0):
        super().__init__()
        assert variant in ("A", "B")
        # R2 — внимание между собранными историями; R1F — его контроль ёмкости
        assert readout in ("R0", "R1", "R2", "R1F")
        self.d, self.stride, self.variant, self.readout = d, stride, variant, readout
        self.window, self.layers = window, layers
        # stem: три соседних кадра одного сустава + их маски, веса общие
        self.stem = nn.Sequential(nn.Linear(3 * (2 * STEM_R + 1) + (2 * STEM_R + 1), d),
                                  nn.GELU(), nn.Linear(d, d))
        self.joint_id = nn.Embedding(NJ, d)
        self.blocks = nn.ModuleList([
            Block(d, heads, window, ffn_mult, dropout,
                  joint_first_layer=(variant == "B")) for _ in range(layers)])
        # Канал таза отдельным токеном: после root-centering его координаты
        # тождественно нулевые, а общая траектория удалена из позных токенов.
        # Ориентация относительной позы одним root-centering не устраняется.
        # Вход открывается ОБУЧАЕМЫМ гейтом, начинающимся у нуля:
        # в монокулярной оценке траектория загрязнена движением камеры
        # и ошибками трекинга, и модель должна иметь право её не брать.
        self.pelvis_dim = pelvis_dim
        n_tokens = NJ
        if pelvis_dim:
            self.pelvis_stem = nn.Sequential(nn.Linear(pelvis_dim, d), nn.GELU(),
                                             nn.Linear(d, d))
            self.pelvis_gate = nn.Parameter(torch.zeros(1))
            n_tokens = NJ + 1
        self.n_tokens = n_tokens
        self.proj_joint = nn.Linear(n_tokens * d, d)    # 23x128 -> 128
        self.proj_time = nn.Linear(stride * d, d)       # 10x128 -> 128
        self.act = nn.GELU()
        self.mix = None
        if readout in ("R2", "R1F"):
            self.mix = HistoryMix(d, heads, ffn_mult, dropout,
                                  mode="attn" if readout == "R2" else "ffn")
        self.out_norm = nn.LayerNorm(d)

    # ------------------------------------------------------------------ stem
    def tokens(self, pos, observed, pelvis=None):
        B, T, J, _ = pos.shape
        # Обнуляем ненаблюдаемые координаты ДО развёртки по времени. Иначе
        # значение ненаблюдённого кадра попадает в окно соседнего НАБЛЮДЁННОГО
        # токена: маска сообщает stem о пропуске, но само число уже подано, а
        # NaN из такого кадра проходит насквозь до выхода.
        pos = torch.where(observed.unsqueeze(-1), pos, torch.zeros_like(pos))
        p = pos.permute(0, 2, 3, 1).reshape(B * J * 3, 1, T)
        p = F.pad(p, (STEM_R, STEM_R)).unfold(2, 2 * STEM_R + 1, 1)
        p = p.view(B, J, 3, T, 2 * STEM_R + 1).permute(0, 3, 1, 4, 2)
        p = p.reshape(B, T, J, 3 * (2 * STEM_R + 1))
        m = observed.permute(0, 2, 1).reshape(B * J, 1, T).float()
        m = F.pad(m, (STEM_R, STEM_R)).unfold(2, 2 * STEM_R + 1, 1)
        m = m.view(B, J, T, 2 * STEM_R + 1).permute(0, 2, 1, 3)
        # Координаты и маски подаются вместе: порядок трёх кадров сохраняется,
        # усреднения перед stem нет.
        h = self.stem(torch.cat([p, m], -1))
        h = h + self.joint_id.weight.view(1, 1, J, -1)
        if self.pelvis_dim:
            if pelvis is None:
                raise ValueError("модель собрана с каналом таза, но он не подан")
            pelvis = torch.where(observed.any(-1, keepdim=True), pelvis,
                                 torch.zeros_like(pelvis))
            g = torch.tanh(self.pelvis_gate)
            pt = self.pelvis_stem(pelvis).unsqueeze(2) * g      # (B,T,1,d)
            h = torch.cat([h, pt], 2)
        return h

    # --------------------------------------------------------------- readout
    def _readout(self, h, valid):
        B, T, J, d = h.shape
        N = T // self.stride
        h = h[:, :N * self.stride].reshape(B, N, self.stride, J, d)
        v = valid[:, :N * self.stride].reshape(B, N, self.stride, J)
        h = h * v.unsqueeze(-1)
        if self.readout == "R0":
            z = self.act(self.proj_joint(h.reshape(B, N, self.stride, J * d)))
            z = self.proj_time(z.reshape(B, N, self.stride * d))
        else:
            z = h.permute(0, 1, 3, 2, 4).reshape(B, N, J, self.stride * d)
            z = self.act(self.proj_time(z))             # (B,N,J,d)
            if self.mix is not None:
                # история сустава пригодна, если он наблюдался хоть в одном
                # кадре окна вывода
                jv = v.any(2).reshape(B * N, J)
                z = self.mix(z.reshape(B * N, J, d), jv).reshape(B, N, J, d)
            z = self.proj_joint(z.reshape(B, N, J * d))
        # Покрытие ЯДРА окна вывода. Полнота истории в halo — отдельная
        # величина: окно нельзя считать качественным только потому, что его
        # ядро заполнено, если контекст был наполовину ненаблюдаем.
        cover = v.float().mean(dim=(2, 3))
        return self.out_norm(z), cover

    def forward(self, pos, observed=None, frame_valid=None, pelvis=None):
        B, T, J, _ = pos.shape
        if observed is None:
            observed = torch.ones(B, T, J, dtype=torch.bool, device=pos.device)
        if frame_valid is not None:
            observed = observed & frame_valid.unsqueeze(-1)
        h = self.tokens(pos, observed, pelvis)
        if self.pelvis_dim:
            # токен таза наблюдаем тогда же, когда наблюдаем кадр
            observed = torch.cat([observed, observed.any(-1, keepdim=True)], -1)
        h = h * observed.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, observed)
            h = h * observed.unsqueeze(-1)
        z, cover = self._readout(h, observed)
        centers = torch.arange(z.shape[1], device=pos.device) * self.stride \
            + self.stride // 2
        return z, centers, cover

    @property
    def receptive_field(self) -> int:
        return self.stride + 2 * (STEM_R + self.layers * self.window)
