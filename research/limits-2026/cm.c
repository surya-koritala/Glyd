/* Adaptive context-mixing code length of a matrix of symbols: the sum, over every bit coded, of -log2 p(bit), with p from a
 * logistic mix of context models that learn as they go. This is what an arithmetic coder driven by the same model spends, to
 * within a few bytes a stream; nothing is encoded or decoded here, and nothing is known in advance (no table is sent: every model
 * starts empty, so the learning cost is in the count).
 *
 * A symbol of nb bits is coded bit by bit in the order given (most informative field first). Each model hashes its context
 * with the bits of the symbol coded so far, up to the last nibble boundary, to a 64-byte line: a check word and 15 counters, the
 * binary tree over the nibble's bits. A counter is a probability (22 bits) and a count (10 bits) and moves by 1/(count + 1.5),
 * the count capped at `limit`. The mixer has a weight set per bit position, trained online on the log loss.
 *
 *   cc -O2 -shared -fPIC -o libcm.so cm.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { M_O0, M_LEFT, M_ABOVE, M_LEFT2, M_LA, M_NBHD, M_COL, M_ROW, M_TILE, M_AUX1, M_AUX2, NM };
#define LINE_BITS 18 /* 2^18 lines of 64 bytes: 16 MB a model */
#define NONE 0xFFFFFu

static float st_tab[4096];
static double rate_tab[1024];
static int ready;

static inline uint64_t h64(uint64_t x) {
  x ^= x >> 31;
  x *= 0x7fb5d329728ea185ULL;
  x ^= x >> 27;
  x *= 0x81dadef4bc2dd44dULL;
  x ^= x >> 33;
  return x;
}

static void init(void) {
  for (int i = 0; i < 4096; i++) {
    double p = (i + 0.5) / 4096.0;
    st_tab[i] = (float)log(p / (1.0 - p));
  }
  for (int n = 0; n < 1024; n++) rate_tab[n] = 1.0 / (n + 1.5);
  ready = 1;
}

/* x: symbols [R*C] (< 2^nb); key: what a neighbour contributes to a context [R*C]; order: bit positions, first coded first;
 * aux1, aux2: per-value contexts or NULL (that model is then off); mask: bit m turns model m on; lr: the mixer's rate;
 * limit: the counters' count cap (< 1024); per_bit (nb doubles, or NULL): bits spent at each coding position.
 * Returns the total bits, or -1 when out of memory. */
double cm_bits(const uint16_t *x, const uint16_t *key, int R, int C, int nb, const int8_t *order, const uint32_t *aux1,
               const uint32_t *aux2, int mask, double lr, int limit, double *per_bit) {
  if (!ready) init();
  if (!aux1) mask &= ~(1 << M_AUX1);
  if (!aux2) mask &= ~(1 << M_AUX2);
  int act[NM], na = 0;
  for (int m = 0; m < NM; m++)
    if (mask >> m & 1) act[na++] = m;
  if (na == 0 || nb < 1 || nb > 16) return -1;
  uint32_t *tab[NM] = {0};
  size_t tbytes = (size_t)64 << LINE_BITS;
  for (int a = 0; a < na; a++) {
    tab[act[a]] = aligned_alloc(64, tbytes);
    if (!tab[act[a]]) {
      for (int b = 0; b < a; b++) free(tab[act[b]]);
      return -1;
    }
    memset(tab[act[a]], 0, tbytes);
  }
  float w[16][NM];
  for (int k = 0; k < 16; k++)
    for (int a = 0; a < na; a++) w[k][a] = (mask & 1) ? (act[a] == M_O0 ? 1.0f : 0.0f) : 1.0f / na;
  if (per_bit) memset(per_bit, 0, sizeof(double) * nb);
  double total = 0.0;
  const float flr = (float)lr;
  uint32_t *line[NM];
  float in[NM];
  for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
      size_t i = (size_t)r * C + c;
      uint64_t L = c > 0 ? key[i - 1] : NONE, U = r > 0 ? key[i - C] : NONE;
      uint64_t base[NM];
      for (int a = 0; a < na; a++) {
        int m = act[a];
        uint64_t v;
        switch (m) {
          case M_O0: v = 0; break;
          case M_LEFT: v = L; break;
          case M_ABOVE: v = U; break;
          case M_LEFT2: v = L | (uint64_t)(c > 1 ? key[i - 2] : NONE) << 20; break;
          case M_LA: v = L | U << 20; break;
          case M_NBHD:
            v = L | U << 20 | (uint64_t)(r > 0 && c > 0 ? key[i - C - 1] : NONE) << 40;
            v = h64(v) ^ (r > 0 && c + 1 < C ? key[i - C + 1] : NONE);
            break;
          case M_COL: v = (uint64_t)c; break;
          case M_ROW: v = (uint64_t)r; break;
          case M_TILE: v = (uint64_t)(r >> 6) << 24 | (uint64_t)(c >> 7); break;
          case M_AUX1: v = aux1[i]; break;
          default: v = aux2[i]; break;
        }
        base[a] = h64(v * 0x9E3779B97F4A7C15ULL + (uint64_t)(m + 1) * 0xBF58476D1CE4E5B9ULL);
      }
      uint32_t sym = x[i], prefix = 0, node = 1;
      for (int k = 0; k < nb; k++) {
        if ((k & 3) == 0) { /* a nibble boundary: each model's line for (its context, the bits so far) */
          for (int a = 0; a < na; a++) {
            uint64_t hh = h64(base[a] ^ ((uint64_t)prefix * 0x94D049BB133111EBULL + ((uint64_t)k << 56)));
            uint32_t *ln = tab[act[a]] + ((hh >> (64 - LINE_BITS)) << 4);
            uint32_t chk = (uint32_t)(hh & 0xFFFFFFFFu) | 1u;
            if (ln[0] != chk) { /* another context had this line: start it afresh */
              ln[0] = chk;
              for (int s = 1; s < 16; s++) ln[s] = 1u << 31; /* p = 1/2, count 0 */
            }
            line[a] = ln;
          }
          node = 1;
        }
        int bit = (sym >> order[k]) & 1;
        float dot = 0.0f;
        for (int a = 0; a < na; a++) {
          in[a] = st_tab[line[a][node] >> 20];
          dot += w[k][a] * in[a];
        }
        double p = 1.0 / (1.0 + exp(-(double)dot));
        if (p < 1e-7) p = 1e-7;
        if (p > 1.0 - 1e-7) p = 1.0 - 1e-7;
        double cost = bit ? -log2(p) : -log2(1.0 - p);
        total += cost;
        if (per_bit) per_bit[k] += cost;
        float err = (float)(bit - p) * flr;
        for (int a = 0; a < na; a++) w[k][a] += err * in[a];
        for (int a = 0; a < na; a++) {
          uint32_t s = line[a][node], n = s & 1023u;
          int64_t p22 = s >> 10;
          p22 += (int64_t)((((int64_t)bit << 22) - p22) * rate_tab[n]);
          if (p22 < 1) p22 = 1;
          if (p22 > (1 << 22) - 1) p22 = (1 << 22) - 1;
          if ((int)n < limit) n++;
          line[a][node] = (uint32_t)p22 << 10 | n;
        }
        node = node * 2 + bit;
        prefix = prefix * 2 + bit;
      }
    }
  }
  for (int a = 0; a < na; a++) free(tab[act[a]]);
  return total;
}
