# ============================================================
# LEARNING CURVE — Random Forest Route Classification
# Paste di cell notebook setelah training RF
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import learning_curve, StratifiedKFold

# ── 1. Hitung Learning Curve ─────────────────────────────────
print("Menghitung learning curve... (bisa 3-5 menit)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    rf,                          # model RF yang sudah dilatih
    X, y,                        # seluruh data (bukan hanya train)
    cv=cv,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    verbose=0
)

# ── 2. Statistik per titik ───────────────────────────────────
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

# ── 3. Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Training line
ax.plot(train_sizes, train_mean,
        'o-', color='#2E75B6', lw=2.5, ms=8,
        label='Training Accuracy')
ax.fill_between(train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.12, color='#2E75B6')

# Validation line
ax.plot(train_sizes, val_mean,
        's--', color='#70AD47', lw=2.5, ms=8,
        label='Validation Accuracy (CV=5)')
ax.fill_between(train_sizes,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.12, color='#70AD47')

# Anotasi gap akhir
gap = train_mean[-1] - val_mean[-1]
ax.annotate(
    f'Gap: {gap:.4f}',
    xy=(train_sizes[-1], (train_mean[-1] + val_mean[-1]) / 2),
    xytext=(-120, 0), textcoords='offset points',
    arrowprops=dict(arrowstyle='->', color='#595959'),
    fontsize=10, color='#595959'
)

# Garis referensi akurasi final
ax.axhline(y=val_mean[-1], color='#70AD47', ls=':', lw=1.2, alpha=0.6)
ax.text(train_sizes[0], val_mean[-1] + 0.003,
        f'Final Val: {val_mean[-1]:.4f}',
        color='#70AD47', fontsize=9)

ax.axhline(y=train_mean[-1], color='#2E75B6', ls=':', lw=1.2, alpha=0.6)
ax.text(train_sizes[0], train_mean[-1] + 0.003,
        f'Final Train: {train_mean[-1]:.4f}',
        color='#2E75B6', fontsize=9)

# Styling
ax.set_xlabel('Jumlah Data Training', fontsize=13)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Learning Curve — Random Forest Route Classification\n'
             'Kabupaten Cilacap Flood Evacuation System',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(train_sizes[0] * 0.95, train_sizes[-1] * 1.02)
ax.set_ylim(0.70, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. Print tabel hasil ─────────────────────────────────────
print("\n" + "="*55)
print("LEARNING CURVE — RINGKASAN HASIL")
print("="*55)
print(f"{'Training Size':>15}  {'Train Acc':>10}  {'Val Acc':>10}  {'Gap':>8}")
print("-"*55)
for ts, tm, vm in zip(train_sizes, train_mean, val_mean):
    print(f"{int(ts):>15,}  {tm:>10.4f}  {vm:>10.4f}  {tm-vm:>8.4f}")

print("\nKesimpulan:")
final_gap = train_mean[-1] - val_mean[-1]
if final_gap < 0.02:
    print(f"  ✅ Gap akhir {final_gap:.4f} < 0.02 → Model TIDAK overfit")
elif final_gap < 0.05:
    print(f"  ⚠️  Gap akhir {final_gap:.4f} → Slight overfit, masih acceptable")
else:
    print(f"  ❌ Gap akhir {final_gap:.4f} > 0.05 → Overfit, perlu tuning")

plateau_start = None
for i in range(1, len(val_mean)):
    if abs(val_mean[i] - val_mean[i-1]) < 0.002:
        plateau_start = train_sizes[i]
        break
if plateau_start:
    print(f"  ✅ Plateau validasi mulai di ~{int(plateau_start):,} sampel")
    print(f"     → Augmentasi {int(train_sizes[-1]):,} sampel sudah cukup")
