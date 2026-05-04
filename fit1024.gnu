set print "zetas.dat"; 
do for[anh in "2 4 6 8 10 12 14 16 18 20"] { 
	f(x)=-(1+2*a)*x+b; 
	fit [:-3] f(x) sprintf('S_avg_L=1024_ANH=%s.txt',anh) u (log($1)):(log($2)) via a,b; 
	p sprintf('S_avg_L=1024_ANH=%s.txt',anh) u (log($1)):(log($2)),f(x); 
	print anh," ", a, " ", (4*anh-1.)/(4*anh-2.), a- (4*anh-1.)/(4*anh-2.); pause 0.1 
}

plot 'zetas.dat' u 1:4 w lp
