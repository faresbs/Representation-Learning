 
#Implementation of the JSD

optimizer.zero_grad()
p = torch.cat((p1,torch.rand(batch_size,1)),1).to(device)
q = torch.cat((phi*torch.ones(batch_size,1),t
                       orch.rand(batch_size,1)),1).to(device)
Dp = net(p)
Dq = net(q)
loss = -(math.log(2.) + (1/2.)*torch.mean(torch.log(Dp)) + 
                                         (1/2.)*torch.mean(torch.log(1-Dq)))
loss.backward()
optimizer.step()
