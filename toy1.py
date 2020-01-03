from seq2seq import Seq2seq

"""
    Toy task 1: memorization (aka overfitting)

    Memorize a small dataset

"""


def main():
    seq2seq = Seq2seq(lr=0.3, init_range=0.3)

    for i in range(1000):
        import random
       
        cost=0
        for t in range(10):
            #a=random.randrange(9)
           
            #b=random.randrange(9)
            cost = seq2seq.train([1,1],[1])
            cost += seq2seq.train([8,1],[1])
            cost += seq2seq.train([7,1],[1])
            cost += seq2seq.train([9,1],[1])
            cost += seq2seq.train([4,1],[1])
            cost += seq2seq.train([3,1],[1])
            cost += seq2seq.train([1,1],[1])
            cost += seq2seq.train([4,1],[1])
            cost += seq2seq.train([0,2],[2])
            cost += seq2seq.train([3,2],[2])
            cost += seq2seq.train([5,2],[2])
            cost += seq2seq.train([6,2],[2])
            cost += seq2seq.train([1,2],[2])
            cost += seq2seq.train([9,2],[2])
            cost += seq2seq.train([8,2],[2])
            cost += seq2seq.train([7,2],[2])
            cost += seq2seq.train([6,2],[2])
            cost += seq2seq.train([5,2],[2])
            cost += seq2seq.train([4,2],[2])
            cost += seq2seq.train([3,2],[2])
            cost += seq2seq.train([2,2],[2])
            cost += seq2seq.train([1,2],[2])
            
        print ('training cost:', cost / 22)
            
     

        if i % 100 == 0:
            print ('Epoch:', i)
            print ('training cost:', cost / 3)
            a=random.randrange(9)
            b=random.randrange(9)

            print ([5, 2], '->', seq2seq.predict([5, 2]))
           # print ([1], '->', seq2seq.predict([1]))
          
      
            
            


if __name__ == "__main__":
    main()
import sys
import os
print(os.getcwd())
print(sys.argv[0])