# Parallel-Monte-Carlo-Integral-Solver
Integral is defined in a text file as explained below.

- The lower integration limits are defined in the first line (for all integration dimensions in order)
- The upper integration limits are defined in the second line (for all integration dimensions in order)
- Each following line represent a term in the integrad polynomial 
	- First number is the coefficient of the term
	- The following numbers are the powers of the variables of each dimension in order

### File example
 
<br />1 4		# lower limts: 	x=1, y=4
<br />2 5		# upper limits: x=2, y=5
<br />0.5 3 7  	# (0.5*x^3*y^7)
<br />0.9 1 6  	# (0.9*x^1*y^6)
