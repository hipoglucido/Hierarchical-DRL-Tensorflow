class A:
    def m1(self):
        print(1)
        
    def m2(self):
        self.m1()
        print(2)
class B(A):
    def m1(self):
        print(3)
        
        
b = B()
b.m2()