class MovingAverage:
		def __init__(self, n):
				self.samples = n
				self.data = []
				self.weights = list(range(1, n+1))

		def add_sample(self, new_sample):
				if len(self.data) < self.samples:
						self.data.append(new_sample)
				else:
						self.data = self.data[1:] + [new_sample]
				#print("samples: %s" % self.data)

		def get_mm(self):
				return float(sum(self.data)) / len(self.data)

		def get_wmm(self):
				s = 0
				for i, x in enumerate(self.data):
						s += x * self.weights[i]
				return float(s) / sum(self.weights[:len(self.data)])

if __name__ == '__main__':
		L = [18, 19,17,19,50,20,19,18]
		mm = MovingAverage(5)
		for x in L:
				print("Adding a sample: %d" % x)
				mm.add_sample(x)
				print("Moving Average: %.2f" % mm.get_mm())
				print("Weighted Moving Average: %.2f" % mm.get_wmm())
				print("")
