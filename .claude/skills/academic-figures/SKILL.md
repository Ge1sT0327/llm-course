---
name: academic-figures
description: Generate publication-quality academic figures (bar, scatter, line, heatmap, box, violin, forest plots) with 300 DPI + SVG output, 4 color palettes, Chinese font support
---

# Academic Figures Skill

Generate publication-quality academic figures from data using Python/matplotlib.

## Chart Types
Bar, grouped bar, stacked bar, scatter (with regression), line (multi-series), heatmap, box, violin, forest plots.

## Color Palettes
- `nature` - Nature journal style (default): #E64B35 #4DBBD5 #00A087 #3C5488 #F39B7F #8491B4
- `lancet` - Lancet journal style: #00468B #ED0000 #42B540 #0099B4 #925E9F #FDAF91
- `conservative` - Muted: #4472C4 #ED7D31 #A5A5A5 #FFC000 #5B9BD5 #70AD47

## Python Template

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# Build your figure here

fig.savefig('output.png', dpi=300, bbox_inches='tight')
fig.savefig('output.svg', format='svg', bbox_inches='tight')
```

## For Architecture/Flow Diagrams
Use `matplotlib.patches` (FancyBboxPatch, FancyArrowPatch) for architecture diagrams, flow charts, and network graphs.
