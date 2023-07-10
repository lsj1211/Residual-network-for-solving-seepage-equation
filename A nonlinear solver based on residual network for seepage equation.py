import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import numpy as np

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

#%% Define Res_Block
class Res_Block(nn.Module):
    def __init__(self, input_unit, hidden_unit):
        super(Res_Block, self).__init__()
        self.input_unit = input_unit
        self.hidden_unit = hidden_unit
        self.layer1 = nn.Linear(input_unit,hidden_unit,bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(input_unit,hidden_unit,bias=False)
        
    def forward(self, inputs):
        x = inputs
        if self.input_unit == self.hidden_unit:
            x = self.layer1(x)
            outputs = self.relu(x)+inputs
        else:
            x = self.layer1(x)
            outputs = self.relu(x) + self.layer2(inputs)
        return outputs

#%% Define residual network
class Res_Net(nn.Module):
    def __init__(self, ResBlock, input_unit, hidden_unit, out_unit):
        super(Res_Net, self).__init__()
        self.layer1=ResBlock(input_unit, hidden_unit)
        self.layer2=nn.Linear(hidden_unit, out_unit, bias=True)
    
    def forward(self, b):
        x=b
        b=self.layer1(b)
        outputs=self.layer2(b) + x
        return outputs

#%% Set well parameters
dt=864.0  # Time step, Unit: s
rw=0.1  # Well radius, Unit: m
S=-0.75  # Skin factor
C=2e-5  # Wellbore storage, Unit: m3/Pa
Nx=5  # Number of grids in x-direction
Ny=5  # Number of grids in y-direction
K=1e-13  # Horizontal permeability, Unit: m2
dx=100.0 # Step x-direction, Unit: m
dy=100.0  # Step y-direction, Unit: m
h=30.5  # Reservoir thickness, Unit: m
mu=0.001  # Viscosity, Unit: Pa.s
Bref=1.0 # Formation volume factor at reference pressure
Pref=3.0e7 # Reference pressure, Unit: Pa
phiref=0.2 # Porosity at reference pressure
Cf=7e-10  # Oil compressibility, Unit: Pa-1
Cr=1e-10  # Rock compressibility, Unit: Pa-1
N=Nx*Ny  # Total number of grids
V=dx*dy*h  # Volume
Q=torch.zeros(N,)
Q[int((N-1)/2)]=1.4e-3  # Constant flow rate, Unit: m3/s

#%% Define system of equations
def F(Pn): # Pn is the pressure to be solved
        F=torch.matmul(T*(torch.reshape((torch.exp(Cf*(Pn-Pref))/Bref).repeat(N),(N,N)).t()),Pn)+(V/dt)*((phiref*torch.exp(Cr*(Pn-Pref))*torch.exp(Cf*(Pn-Pref)))-(phiref*torch.exp(Cr*(P-Pref))*torch.exp(Cf*(P-Pref))))/Bref+q
        return F

#%% Define system of equations
def F1(Pn): # Pn is the pressure to be solved
        F1=torch.matmul(T*(torch.reshape((torch.exp(Cf*(Pn[0,:]-Pref))/Bref).repeat(N),(N,N)).t()),Pn[0,:])+(V/dt)*((phiref*torch.exp(Cr*(Pn[0,:]-Pref))*torch.exp(Cf*(Pn[0,:]-Pref)))-(phiref*torch.exp(Cr*(P-Pref))*torch.exp(Cf*(P-Pref))))/Bref+q
        F1=torch.reshape(F1,(1,N))
        return F1

#%% Initialisation
P=3.0e7*torch.ones(N,) # Initial pressure, Unit: Pa
P=P.to(device)
Pw=P[int((N-1)/2)]
q=torch.zeros(N,)
q=q.to(device)
Thalf=(K*dy*h)/(mu*dx)

#%% Define transmissibility coeffcient matrix
T=torch.zeros(N,N)
for i in range(0,N):
    if (i+1) % Nx != 1:  # Not at the left boundary
        T[i,i-1] = - Thalf
        T[i,i]   = T[i,i] - T[i,i-1]
    else:  # At the left boundary
        T[i,i] = T[i,i]

    if (i+1) % Nx != 0:  # Not at the right boundary
        T[i,i+1] = - Thalf
        T[i,i]   = T[i,i] - T[i,i+1]
    else:  # At the right boundary
        T[i,i] = T[i,i]

    if int(i / Nx) > 0:  # Not at bottom boundary
        T[i,i-Nx] = - Thalf
        T[i,i]   = T[i,i] - T[i,i-Nx]
    else:  # At bottom boundary
        T[i,i] = T[i,i]

    if int(i / Nx) < Ny - 1:  # Not at top boundary
        T[i,i+Nx] = - Thalf
        T[i,i]   = T[i,i] - T[i,i+Nx]
    else:  # At top boundary
        T[i,i] = T[i,i]
T=T.to(device)

#%% Fix the random numbers.
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
# Set the random numbers seed
setup_seed(10)

#%% Set the sample size, number of neurons in each layer
data_num = 8000 # sample size
input_neuron = N # Number of neurons in the input layer
hidden_neuron = 300 # Number of neurons in the hidden layer
output_neuron = N # Number of neurons in the output layer

#%% Define network, loss and optimizer
model=Res_Net(Res_Block, input_neuron, hidden_neuron, output_neuron)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1)
model = model.to(device)

#%% Define dataset and label
x=torch.normal(2.506e7,2.506e7,size=(data_num,input_neuron))
x=x.to(device)
Input=torch.zeros(input_neuron,data_num) # Input sample matrix, one sample per column
Input=Input.to(device)
for i in range(data_num):
    Input[:,i]=F(x[i,:])
Target=x
Input=Input.t()
Target=Target.to(device)
Input=Input.to(device)

#%% Train
start_train = time()
print('Training...')
for epoch in range(30000):
    output=model(Input)
    loss = criterion(output, Target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch%100==0):
         print('The '+str(epoch)+'th loss is:'+str(loss.data))
end_train = time()
print("The train time is:",end_train-start_train)

#%% Test
print('Test...')
with torch.no_grad():
    Input_ceshi=torch.zeros(1,N)
    Input_ceshi=Input_ceshi.to(device)
    output_ceshi=model(Input_ceshi)

start_test = time()
Pw_all=np.array([]) # Store the BHP at different time steps
with torch.no_grad():
    for t in range(3000):
        #%% Correction iteration method
        print('Correction iteration...')
        for times in range(5):
            error_old=torch.abs(torch.norm(F1(output_ceshi)-Input_ceshi))
            Input1 = Input_ceshi.clone()
            for times in range(10):
                m=F1(output_ceshi)
                Input1=Input1-m
                a=1e8/Input1.norm()
                output_new=output_ceshi + model(Input1*a)/a
                error_new=torch.abs(torch.norm(F1(output_new)-Input_ceshi))
                if (error_new/error_old)>1.1:
                    break
                output_ceshi=output_new
                error_old=error_new
        print('The error is:', error_old)

        #%% Calculate the BHP
        Pn=output_ceshi[0,:]
        P=Pn
        P=P.to(device)
        Jw=(2.0*torch.pi*K*h)/(mu*(Bref/(torch.exp(Cf*(P[int((N-1)/2)]-Pref))))*(torch.log(torch.tensor(0.2*dx/rw))+S))
        Pw=(Jw*P[int((N-1)/2)]+(C/dt)*Pw-Q[int((N-1)/2)])/(Jw+(C/dt))
        q[int((N-1)/2)]=Jw*(P[int((N-1)/2)]-Pw)
        q=q.to(device)
        print('The BHP at the %d time step is %f:' % (t+1,Pw))

        Pw_new=Pw
        Pw_new=Pw_new.detach().cpu().numpy()
        Pw_all=np.append(Pw_all,Pw_new)
end_test = time()
print("The test time is:",end_test-start_test)
print("The BHP at different time steps:",Pw_all)
np.savetxt("Pw_all.txt",Pw_all,fmt='%.6f')