set terminal pdf
set output 'rural.pdf'
set title "Rural scenario"

#set logscale x

data_folder='../results/'

plot data_folder.'rural-data.gnuplot' i 0 u 1:2 w boxes title "Empirical PDF", '' i 1 u 1:2 w lines title "fitted PDF"

set output 'suburban.pdf'
set title "Suburban scenario"
plot data_folder.'suburban-data.gnuplot' i 0 u 1:2 w boxes title "Empirical PDF", '' i 1 u 1:2 w lines title "fitted PDF"

set output 'urban.pdf'
set title "Urban scenario"
plot data_folder.'urban-data.gnuplot' i 0 u 1:2 w boxes title "Empirical PDF", '' i 1 u 1:2 w lines title "fitted PDF"

set style fill transparent solid 0.2 noborder
set output 'summary.pdf'

set title 'Fitted PDF for all the Scenarios'
plot data_folder.'rural-data.gnuplot' i 0 u 1:2 w boxes lc rgb "#00FF0000" title '',\
     data_folder.'suburban-data.gnuplot' i 0  u 1:2 w boxes lc rgb "#000000FF" title '',\
     data_folder.'urban-data.gnuplot' i 0 u 1:2 w boxes lc rgb "#0000FF00" title '',\
     data_folder.'rural-data.gnuplot' i 1 u 1:2 w lines title "Rural Areas" lc "red" lw 4, \
     data_folder.'suburban-data.gnuplot' i 1 u 1:2 w lines title "Suburban Areas" lc "blue" lw 4,\
     data_folder.'urban-data.gnuplot' i 1 u 1:2 w lines title "Urban Areas" lc "green" lw 4, \
