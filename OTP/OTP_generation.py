import random

alphabet='abcdefghijklmnopqrstuvwxyz'
a=[]
def generate(num_lett):
    for i in range(num_lett):
        random_letter=random.choice(alphabet)
        a.append(random_letter)
    
    otp=''.join(a)
    print((otp))
    
num_lett=int(input ("Whats length of  OTP you want to Generate?"))
generate(num_lett)


    