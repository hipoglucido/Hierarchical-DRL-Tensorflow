from decimal import *


def common_d(x1, x2, start, end, increment):
	getcontext().prec = 3
	common_ds = []
	d = Decimal(start)
	while d < end:
		y1 = float(x1) / float(d)
		y2 = float(x2) / float(d)
#		print("y1", y1, "y2", y2)
		if round(y1) == y1 and round(y2) == y2:
			common_ds += [d]
		d += Decimal(increment)
		print(d)
	return common_ds

print(common_d(448, 448, 2, 10, 0.01))