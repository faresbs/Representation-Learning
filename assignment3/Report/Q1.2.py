 
#Implementation of the WD

optimizer.zero_grad()
p = torch.cat((p1,torch.rand(batch_size,1)),1).to(device)
q = torch.cat((phi*torch.ones(batch_size,1),
                       torch.rand(batch_size,1)),1).to(device)
Dp = net(p)
Dq = net(q)
#  gradient penalty
a = torch.rand(batch_size, 1).expand(batch_size,2).to(device)
r = a*p + (1-a)*q
r.requires_grad = True
Dr = net(r)
gradients = torch.autograd.grad(outputs=Dr, inputs=r,
                grad_outputs=torch.ones(batch_size,1).to(device),
                create_graph=True, retain_graph=True, only_inputs=True)[0]
loss = -(torch.mean(Dp) - torch.mean(Dq) - 
         gp_coeff*torch.mean((gradients.norm(2, dim=1) - 1) ** 2))
loss.backward()
optimizer.step()