import numpy as np

from layers.util import tanh, sigmoid, sigmoid_grad, tanh_grad
from layers.util import initalize, zeros, ones
from layers.util import softmax_grad

class Lstm(object):

    def __init__(self, input_size, hidden_size, init_range=1.0, previous=None):
        self.input_size, self.hidden_size = input_size, hidden_size

        if previous:
            self.previous = previous
            previous.next = self

        # initalize weights
        def init(x,y):
            return initalize((x,y), init_range)

        h, n = hidden_size, input_size

        self.W_hi, self.W_hf, self.W_ho, self.W_hj = init(h, h), init(h, h), init(h, h), init(h, h)
        self.W_xi, self.W_xf, self.W_xo, self.W_xj = init(h, n), init(h, n), init(h, n), init(h, n)
        
        
        self.W_ci, self.W_cf, self.W_co, self.W_cj = init(h, n), init(h, n), init(h, n), init(h, n)
        self.W_av, self.W_aa, self.W_ua = init(h, h), init(h, h), init(h, h)
        
        self.b_i, self.b_f, self.b_o, self.b_j = zeros(h), ones(h), zeros(h), zeros(h)
        #initalize the attention vector
        self.W_av,self.W_aa,self.W_aua = init(h, h), [init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h)], [init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h),init(h, h)]

        
        # initalize gradients
        self.dW_hi, self.dW_hf, self.dW_ho, self.dW_hj = zeros(h, h), zeros(h, h), zeros(h, h), zeros(h, h)
        self.dW_xi, self.dW_xf, self.dW_xo, self.dW_xj = zeros(h, n), zeros(h, n), zeros(h, n), zeros(h, n)
        self.dW_ci, self.dW_cf, self.dW_co, self.dW_cj = zeros(h, h), zeros(h, h), zeros(h, h), zeros(h, h)
        
        self.dW_av,self.dW_aa,self.dW_ua = zeros(h, h), zeros(h, h), zeros(h, h)
        self.db_i, self.db_f, self.db_o, self.db_j = zeros(h), zeros(h), zeros(h), zeros(h)

        # list of all parameters
        self.params = [
            ('W_hi', self.W_hi, self.dW_hi),
            ('W_hf', self.W_hf, self.dW_hf),
            ('W_ho', self.W_ho, self.dW_ho),
            ('W_hj', self.W_hj, self.dW_hj),

            ('W_xi', self.W_xi, self.dW_xi),
            ('W_xf', self.W_xf, self.dW_xf),
            ('W_xo', self.W_xo, self.dW_xo),
            ('W_xj', self.W_xj, self.dW_xj),
            
            ('W_ci', self.W_ci, self.dW_ci),
            ('W_cf', self.W_cf, self.dW_cf),
            ('W_co', self.W_co, self.dW_co),
            ('W_cj', self.W_cj, self.dW_cj),
            
            ('W_av', self.W_av, self.dW_av),
            ('W_aa', self.W_aa, self.dW_aa),
            ('W_aua', self.W_ua, self.dW_ua),
            

            ('b_i', self.b_i, self.db_i),
            ('b_f', self.b_f, self.db_f),
            ('b_o', self.b_o, self.db_o),
            ('b_j', self.b_j, self.db_j),
            
        ]

        self.initSequence()

    def initSequence(self):
        self.t = 0
        self.tc = 0
        self.x = {}
        self.xa = {}
        self.h = {}
        self.hidden = {}
        self.context = {}
        self.e_i_j_t = {}
        self.e_i_j = {}
        self.s = {}
        self.c = {}
        self.ct = {}

        self.input_gate = {}
        self.forget_gate = {}
        self.output_gate = {}
        self.cell_update = {}
        self.attention=[]


        if hasattr(self, 'previous'):
            self.h[0] = self.previous.h[self.previous.t]
            self.c[0] = self.previous.c[self.previous.t]
        else:
            self.h[0] = zeros(self.hidden_size)
            self.c[0] = zeros(self.hidden_size)

        if hasattr(self, 'next'):
            self.dh_prev = self.next.dh_prev
            self.dc_prev = self.next.dc_prev
        else:
            self.dh_prev = zeros(self.hidden_size)
            self.dc_prev = zeros(self.hidden_size)

        # reset all gradients to zero
        for name, param, grad in self.params:
            grad[:] = 0

    def forward(self, x_t,hidden):
        
        
            
        if hasattr(self, 'previous'):

            self.tc += 1
            t = self.tc
            if t==1:
                self.s[t-1]=hidden[(len(hidden)-1)]
             
    
                
            ##注意力模型计算context向量
            vect_e=[]                                                                                                                                                                                                                                                                                                                                          
            sum_e=0
            
            for j in range(len(hidden)):
              
           
                self.e_i_j_t[j]=tanh(np.dot(self.W_aa[j],self.s[t-1])+np.dot(self.W_ua[j],hidden[j]))
         
                self.e_i_j[j] = np.dot(self.W_av[j].T,tanh(np.dot(self.W_aa[j],self.s[t-1])+np.dot(self.W_ua[j],hidden[j])))
                vect_e.append(self.e_i_j[j])
                sum_e+=self.e_i_j[j]
            attention=[]
            for e_i_j in range(len(hidden)):
                vect_a=vect_e[e_i_j]/sum_e
    
               
              
                attention.append(vect_a)
              
            self.attention.append(attention)
            attention = np.asarray(self.attention)
            try:
                hidden[t-1]=np.array(hidden[t-1])
            except IndexError:
                print(t)
                print(len(hidden))
                print(hidden)
               
       
            context=[0 for i in range(10)]
            #for i in range(len(hidden)):
            attention=attention.ravel()
            for num in range(len(hidden)):
                temp=[]
                for i in range(len(hidden[num])):
                    temp.append(float(hidden[num][i]*attention[num])) 

                context=np.sum([temp,context], axis = 0)
    
          
            context= context.tolist()
      
            self.context[t-1]=context
                    
    
                    
                
            ##注意力模型计算context向量
               
            self.input_gate[t] = sigmoid(np.dot(self.W_hi, x_t) + np.dot(self.W_ci, context)+ np.dot(self.W_xi, self.s[t-1]) + self.b_i)
            self.forget_gate[t] = sigmoid(np.dot(self.W_hf, x_t) + np.dot(self.W_cf, context)+ np.dot(self.W_xf, self.s[t-1]) + self.b_f)
            self.output_gate[t] = sigmoid(np.dot(self.W_ho, x_t) + np.dot(self.W_co, context)+ np.dot(self.W_xo, self.s[t-1]) + self.b_o)
            self.cell_update[t] = tanh(np.dot(self.W_hj, x_t) + np.dot(self.W_cj, context)+ np.dot(self.W_xj, self.s[t-1]) + self.b_j)
            
            self.c[t] = self.input_gate[t] * self.cell_update[t] + self.forget_gate[t] * self.c[t-1]
            self.ct[t] = tanh(self.c[t])
            self.s[t] = self.output_gate[t] * self.ct[t]
       
            self.x[t] = x_t
            return self.s[t]
            
            
            
        else:
            
            self.t += 1
    
            t = self.t
            h = self.h[t-1]
    
            self.input_gate[t] = sigmoid(np.dot(self.W_hi, x_t) + np.dot(self.W_xi, h) + self.b_i)
            self.forget_gate[t] = sigmoid(np.dot(self.W_hf, x_t) + np.dot(self.W_xf, h) + self.b_f)
            self.output_gate[t] = sigmoid(np.dot(self.W_ho, x_t) + np.dot(self.W_xo, h) + self.b_o)
            self.cell_update[t] = tanh(np.dot(self.W_hj, x_t) + np.dot(self.W_xj, h) + self.b_j)
    
            self.c[t] = self.input_gate[t] * self.cell_update[t] + self.forget_gate[t] * self.c[t-1]
            self.ct[t] = tanh(self.c[t])
            self.h[t] = self.output_gate[t] * self.ct[t]
            self.hidden[t] = self.output_gate[t] * self.ct[t]
            
            
            
            self.x[t] = x_t
            return self.h[t]


            

        
        

    def backward(self, dh,hidden):
        if hasattr(self, 'previous'):
            t = self.tc
        
        else:
            t = self.t
        

        dh = dh + self.dh_prev
        dC = tanh_grad(self.ct[t]) * self.output_gate[t] * dh + self.dc_prev

        d_input = sigmoid_grad(self.input_gate[t]) * self.cell_update[t] * dC
        d_forget = sigmoid_grad(self.forget_gate[t]) * self.c[t-1] * dC
        d_output = sigmoid_grad(self.output_gate[t]) * self.ct[t] * dh
        d_update = tanh_grad(self.cell_update[t]) * self.input_gate[t] * dC
         
  
        if hasattr(self, 'previous'):
            d_e_i_j = []
           
            
        
        

        self.dc_prev = self.forget_gate[t] * dC

        self.db_i += d_input
        self.db_f += d_forget
        self.db_o += d_output
        self.db_j += d_update
        
        if hasattr(self, 'previous'):
          
            h_in = hidden[t-1]
            c_in = self.context[t-1]
            s_in = self.s[t-1]
         
        else:
            h_in = self.h[t-1]
        
        
        
        
        self.dW_xi += np.outer(d_input, self.x[t])
        self.dW_xf += np.outer(d_forget, self.x[t])
        self.dW_xo += np.outer(d_output, self.x[t])
        self.dW_xj += np.outer(d_update, self.x[t])

        self.dW_hi += np.outer(d_input, h_in)
        self.dW_hf += np.outer(d_forget, h_in)
        self.dW_ho += np.outer(d_output, h_in)
        self.dW_hj += np.outer(d_update, h_in)
        if hasattr(self, 'previous'):
            self.dW_ci += np.outer(d_input, c_in)
            self.dW_cf += np.outer(d_forget, c_in)
            self.dW_co += np.outer(d_output, c_in)
            self.dW_cj += np.outer(d_update, c_in)
         
     
            for j in range(len(self.h)):
                self.dW_aa[j]=np.dot(d_update,self.W_cj)
                self.dW_aa[j]=np.dot(self.dW_aa[j],self.h[j])
                temp = softmax_grad(self.attention[t-1][j])
                self.dW_aa[j]=np.dot(self.dW_aa[j],temp)
                self.dW_aa[j]=np.dot(self.dW_aa[j],self.W_av[j])
                temp_e_i_j =[ x-x*x  for x in self.e_i_j_t[j]]
                self.dW_aa[j]=np.dot(self.dW_aa[j],temp_e_i_j)
                d_e_i_j.append(self.dW_aa[j]) 
                self.dW_aa[j] = np.dot(self.dW_aa[j],s_in[t-1])
  
              
                self.dW_ua[j] = np.dot(d_e_i_j[j],h_in[j])

              
                self.dW_av[j]=np.dot(d_update,self.W_cj)
                self.dW_av[j]=np.dot(self.dW_av[j],self.h[j])
                temp = softmax_grad(self.attention[t-1][j])
                self.dW_av[j]=np.dot(self.dW_av[j],temp)
                self.dW_av[j]=np.dot(self.dW_av[j],self.e_i_j_t[j])
                
                
                
              
          
                
            
            
           

        self.dh_prev = np.dot(self.W_hi.T, d_input)
      
        self.dh_prev += np.dot(self.W_hf.T, d_forget) 
        self.dh_prev += np.dot(self.W_ho.T, d_output)
        self.dh_prev += np.dot(self.W_hj.T, d_update)
        if hasattr(self, 'previous'):
            for j in range(len(self.h)):
            
                self.dh_prev += np.dot(d_e_i_j[j], self.h[j])
            

        dX = np.dot(self.W_xi.T, d_input)
        dX += np.dot(self.W_xf.T, d_forget)
        dX += np.dot(self.W_xo.T, d_output)
        dX += np.dot(self.W_xj.T, d_update)

        self.t -= 1

        return dX

    
