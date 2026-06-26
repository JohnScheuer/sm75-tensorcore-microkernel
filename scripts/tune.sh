gnuplot <<EOF
set datafile separator ","
set terminal png size 900,600
set output "heatmap.png"

set title "Performance Heatmap"
set xlabel "WARPS"
set ylabel "STAGE_K"
set grid

set pm3d map
splot "tuning_results.csv" using 1:2:4 notitle
EOF