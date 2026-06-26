#!/bin/bash

CSV="tuning_results.csv"

if [ ! -f "$CSV" ]; then
  echo "ERROR: tuning_results.csv not found."
  echo "Run ./scripts/tune.sh first."
  exit 1
fi

echo "Generating plots from $CSV ..."

# --------------------------------------------------
# 1️⃣ GFLOPs vs Configuration Index
# --------------------------------------------------
gnuplot <<EOF
set datafile separator ","
set terminal png size 1000,600
set output "plot_overview.png"

set title "Auto-Tuning Overview (GFLOPs)"
set xlabel "Configuration Index"
set ylabel "GFLOPs/s"
set grid
set key left top

plot "$CSV" using 4 with linespoints lw 2 pt 7 title "GFLOPs"
EOF

# --------------------------------------------------
# 2️⃣ GFLOPs vs STAGE_K
# --------------------------------------------------
gnuplot <<EOF
set datafile separator ","
set terminal png size 1000,600
set output "plot_stage_k.png"

set title "GFLOPs vs STAGE_K"
set xlabel "STAGE_K"
set ylabel "GFLOPs/s"
set grid
set key left top

plot "$CSV" using 2:4 with points pt 7 ps 1.5 title "Performance"
EOF

# --------------------------------------------------
# 3️⃣ Heatmap WARPS × STAGE_K
# --------------------------------------------------
gnuplot <<EOF
set datafile separator ","
set terminal png size 1000,600
set output "plot_heatmap.png"

set title "Performance Heatmap (WARPS vs STAGE_K)"
set xlabel "WARPS"
set ylabel "STAGE_K"
set grid
set view map
set pm3d at b
set palette rgb 33,13,10

splot "$CSV" using 1:2:4 notitle
EOF

echo "✅ Plots generated:"
echo "   plot_overview.png"
echo "   plot_stage_k.png"
echo "   plot_heatmap.png"