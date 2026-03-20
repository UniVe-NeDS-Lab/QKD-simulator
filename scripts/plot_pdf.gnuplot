set terminal pdf
set output 'rural.pdf'
set title "Rural scenario"

set logscale x
#set logscale y

data_folder='../results/'
figures_folder='../figures/'

set style fill transparent solid 0.2 noborder
set output figures_folder.'summary.pdf'

set xrange [50:]
set title 'Link Length PDF'
plot data_folder.'length-rural-30-10000-data.gnuplot' i 0 u 1:2 w boxes lc rgb "#00FF0000" title '',\
     data_folder.'length-suburban-30-10000-data.gnuplot' i 0  u 1:2 w boxes lc rgb "#000000FF" title '',\
     data_folder.'length-urban-30-10000-data.gnuplot' i 0 u 1:2 w boxes lc rgb "#0000FF00" title '',\
     data_folder.'length-rural-30-10000-data.gnuplot' i 1 u 1:2 w lines title "Rural Areas" lc "red" lw 4, \
     data_folder.'length-suburban-30-10000-data.gnuplot' i 1 u 1:2 w lines title "Suburban Areas" lc "blue" lw 4,\
     data_folder.'length-urban-30-10000-data.gnuplot' i 1 u 1:2 w lines title "Urban Areas" lc "green" lw 4, \

set output figures_folder.'cdf.pdf'
set title 'Link Length CDF'
plot data_folder.'length-rural-30-10000-data.gnuplot' i 0 u 1:3 w lines title "Rural Areas" lc "red" lw 4, \
     data_folder.'length-suburban-30-10000-data.gnuplot' i 0 u 1:3 w lines title "Suburban Areas" lc "blue" lw 4,\
     data_folder.'length-urban-30-10000-data.gnuplot' i 0 u 1:3 w lines title "Urban Areas" lc "green" lw 4



set output figures_folder.'summary-rate.pdf'

set logscale y
set autoscale x
set xrange [1:500]
set title 'Link Rate PDF'
#plot data_folder.'rate-rural-30-1000-1000000000-data.gnuplot' i 0 u 1:2 w boxes lc rgb "#00FF0000" title '',\
#     data_folder.'rate-suburban-30-1000-1000000000-data.gnuplot' i 0  u 1:2 w boxes lc rgb "#000000FF" title '',\
#     data_folder.'rate-urban-30-1000-1000000000-data.gnuplot' i 0 u 1:2 w boxes lc rgb "#0000FF00" title '',\
#     data_folder.'rate-rural-30-1000-1000000000-data.gnuplot' i 1 u 1:2 w lines title "Rural Areas" lc "red" lw 4, \
#     data_folder.'rate-suburban-30-1000-1000000000-data.gnuplot' i 1 u 1:2 w lines title "Suburban Areas" lc "blue" lw 4,\
#     data_folder.'rate-urban-30-1000-1000000000-data.gnuplot' i 1 u 1:2 w lines title "Urban Areas" lc "green" lw 4, \
#
plot data_folder.'rate-urban-30-1000-1000000000-data.gnuplot' i 0 u 1:2 w boxes lc rgb "#0000FF00" title '',\
     data_folder.'rate-urban-30-1000-1000000000-data.gnuplot' i 1 u 1:2 w lines title "Urban Areas" lc "green" lw 4, \



#set output 'keyrate-summary.pdf'
#plot data_folder.'rural-data.gnuplot' i 2 u 1:2 w  boxes t 'rural',\
#     data_folder.'suburban-data.gnuplot' i 2 u 2:3 w l title 'suburban',\
#     data_folder.'urban-data.gnuplot' i 2 u 2:3 w l title 'urban'

#plot data_folder.'length-rural-30-1000-data.gnuplot' i 0 u 1:2 w boxes title "Empirical PDF", '' i 1 u 1:2 w lines title "fitted PDF"
#
#set output 'suburban.pdf'
#set title "Suburban scenario"
#plot data_folder.'length-suburban-30-1000-data.gnuplot' i 0 u 1:2 w boxes title "Empirical PDF", '' i 1 u 1:2 w lines title "fitted PDF"
#
#set output 'urban.pdf'
#set title "Urban scenario"
#plot data_folder.'length-urban-30-1000-data.gnuplot' i 0 u 1:2 w boxes title "Empirical PDF", '' i 1 u 1:2 w lines title "fitted PDF"


