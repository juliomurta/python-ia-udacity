
class MyMagicMethods():
    def __init__(self, value):
        self.value = value
    def __add__(self, other):
        return MyMagicMethods(self.value + other.value)
    def __repr__(self):
        return "My value is {}.".format(self.value)
       
a = MyMagicMethods(5)
b = MyMagicMethods(10)

c = a + b 

print(c)       
    
    