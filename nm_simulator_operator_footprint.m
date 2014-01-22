% Hardware Simulator for
% Ristricted Boltzmann Machine in Neuron Machine Hardware Architecture
% Code provided by Jerry B. Ahn
clear all; close all;
maxepoch=10; Nn(2)=500; Nn(3)=500; Nn(4)=2000; 
% converter; % converting Raw files into Matlab format
makebatches; % needs converter.m, makebatches.m from [], MNIST files []
[Nc Nn(1) Nb]=size(batchdata);
% Initializing symmetric weights
vishid = 0.1*randn(2000,2000);  % same weights for all layer (for debug)
% vishid = 0.1*randn(Nn(1), Nn(2));
epsilonw = 0.1; epsilonb = 0.1; % Learning rates
fgr=figure('Color',[1 1 1]); %set(fgr, 'Position', [0 0 3000 475]);

fprintf(1,'Setting dual-port block memories and building operation codes...\n');
MW = zeros(60000,64); Mdw = zeros(10000,64); % memories for synaptic data
MM = MW; MR = MW; Mnull = MW + 1;                         % network data
MX = zeros(10000,64); MB = zeros(1,20000); MAp = MB;      % neuronal data

Mdebug1 = MB; Mdebug2 = MB; Mnet = MB; OP = zeros(20,12);

moffs = 0; woffs = 0;roffs = 0; OP(1,1)=1;OP(1,8)=Nn(1);op_i = 2;
for l=1:3
  h_bpn = ceil(Nn(l)/64); hNb = h_bpn*Nn(l+1);
  v_bpn = ceil(Nn(l+1)/64); vNb = v_bpn*Nn(l);
  for j=0:Nn(l+1)-1
    for i=0:Nn(l)-1
      hid_inx = mod(i + j, h_bpn*64);
      hid_bi = floor(hid_inx/64);
      snu_i = mod(hid_inx, 64);
      maddr = j * h_bpn + hid_bi + moffs + 1;
      waddr = j * h_bpn + hid_bi + woffs + 1;
      MW(waddr,snu_i+1) = vishid(i+1,j+1);
      MM(maddr,snu_i+1) = i;
      Mnull(maddr,snu_i+1) = 0;
      vis_inx = mod(i + j, v_bpn*64);
      assert(mod(hid_inx, 64)==mod(vis_inx, 64)); % ensures co-location
      vis_bi = floor(vis_inx/64);
      maddr =  i * v_bpn + vis_bi + hNb + moffs +1;
      raddr =  i * v_bpn + vis_bi + roffs + 1;
      MR(raddr,snu_i+1) = j * h_bpn + hid_bi;
      MM(maddr,snu_i+1) = j;
      Mnull(maddr,snu_i+1) = 0;
    end
  end
  of1 = mod(l,2)*2000;of2=mod(l+1,2)*2000;bo=(l-1)*4000;
  OP(op_i+0,:)=[2,h_bpn,hNb,moffs,Nn(l+1),of1,bo+2000,0,of2,woffs,0,l*4000];
  OP(op_i+1,:)=[3,v_bpn,vNb,moffs+hNb,Nn(l),4000,bo,0,of1,woffs,roffs,0];
  OP(op_i+2,:)=[4,h_bpn,hNb,moffs,Nn(l+1),0,bo+2000,0,4000,woffs,0,0];
  op_i = op_i + 3;
    
  moffs =  moffs + hNb + vNb;
  woffs =  woffs + hNb;
  roffs =  roffs + vNb;
end
OP(op_i-1,8)=Nn(1);OP(op_i,1:2)=[9,2];

% %        [ 1,  2,    3,    4,   5,   6,    7,   8,   9,   10,  11,   12]
% %        [Ty,BPN,SBnum,SBoff, NRn,XWof, Boff,In_n,XRof, WWof, Rof,   BW]
% OP(1,:) = [1,  0,    0,    0,   0,   0,    0, 784,   0,    0,    0,    0];
% OP(2,:) = [2, 13, 6500,    0, 500,2000, 2000,   0,   0,    0,    0, 4000];
% OP(3,:) = [3,  8, 6272, 6500, 784,4000,    0,   0,2000,    0,    0,    0];
% OP(4,:) = [4, 13, 6500,    0, 500,   0, 2000,   0,4000,    0,    0,    0];
% OP(5,:) = [2,  8, 4000,12772, 500,   0, 6000,   0,2000, 6500,    0, 8000];
% OP(6,:) = [3,  8, 4000,16772, 500,4000, 4000,   0,   0, 6500, 6272,    0];
% OP(7,:) = [4,  8, 4000,12772, 500,   0, 6000,   0,4000, 6500,    0,    0];
% OP(8,:) = [2,  8,16000,20772,2000,2000,10000,   0,   0,10500,    0,12000];
% OP(9,:) = [3, 32,16000,36772, 500,4000, 8000,   0,2000,10500,10272,    0];
% OP(10,:)= [4,  8,16000,20772,2000,   0,10000, 784,4000,10500,    0,    0];
% OP(11,:)= [9,  2,    0,0,0,0,0,0,0,0,0,0]; % goto statement

% define control signals, registers, and arithmetic operators
CT=uint16(zeros(1,300)); Csb=CT; Csben=CT; Cnr=CT; Cnren=CT; Q64=zeros(10,200,64);
R=zeros(20,60); R64=zeros(20,130,64); COM=zeros(20,10); C64=zeros(20,10,64);
DspRes = 1; slot_count = zeros(1,10);
Ft=zeros(400,400*DspRes); view=zeros(330,1000*DspRes);

cp=1; ncp=1; cp_cnt = 0; dp_i=0; d_i=0; net_sum = 0; % initialize variables
sbase = 0; s_en = 0; nrsel = 0; nren = 0; done = 0; err = 0;
cmap = lines(8); cmap(1,:) = 1;
fprintf(1,'Starting machine...\n');

myrand = rand(1,Nn(2));
poshidprobs = zeros(Nc,Nn(2));
neghidprobs = zeros(Nc,Nn(2));
posprods    = zeros(Nn(1),Nn(2));
negprods    = zeros(Nn(1),Nn(2));
vishidinc  = zeros(Nn(1),Nn(2));
hidbiasinc = zeros(1,Nn(2));
visbiasinc = zeros(1,Nn(1));
data2=zeros(1,Nn(2));
poshidstates = zeros(Nc, Nn(2));
hidbiases  = zeros(1,Nn(2));
visbiases  = zeros(1,Nn(1));
vishid_1 = vishid(1:Nn(1),1:Nn(2));

myrand2 = rand(1,Nn(3));
poshidprobs2 = zeros(Nc,Nn(3));
neghidprobs2 = zeros(Nc,Nn(3));
posprods2    = zeros(Nn(2),Nn(3));
negprods2    = zeros(Nn(2),Nn(3));
vishidinc2  = zeros(Nn(2),Nn(3));
hidbiasinc2 = zeros(1,Nn(3));
visbiasinc2 = zeros(1,Nn(2));
data3=zeros(1,Nn(3));
poshidstates2 = zeros(Nc, Nn(3));
hidbiases2  = zeros(1,Nn(3));
visbiases2  = zeros(1,Nn(2));
vishid_2 = vishid(1:Nn(2),1:Nn(3));
tempweight = zeros(Nn(2),Nn(3));

myrand3 = rand(1,Nn(4));
poshidprobs3 = zeros(Nc,Nn(4));
neghidprobs3 = zeros(Nc,Nn(4));
posprods3    = zeros(Nn(3),Nn(4));
negprods3    = zeros(Nn(3),Nn(4));
vishidinc3  = zeros(Nn(3),Nn(4));
hidbiasinc3 = zeros(1,Nn(4));
visbiasinc3 = zeros(1,Nn(3));
data4=zeros(1,Nn(4));
poshidstates3 = zeros(Nc, Nn(4));
hidbiases3  = zeros(1,Nn(4));
visbiases3  = zeros(1,Nn(3));
vishid_3 = vishid(1:Nn(3),1:Nn(4));
t1 = zeros(2000,2000); tempweight1 = t1; tempweight2 = t1; tempweight3 = t1;

for ck=1:1E9     % clock cycles
  cPh = OP(cp,1); nPh = OP(ncp,1); ck6 = mod(ck-1+6,400*DspRes)+1; % shortcuts

%   [ck cp ncp sbase delaycnt]
  % Network Unit
  
  sbsel = sbase+OP(cp,4)+1; 
  for k=1:64    % loop(k) functions kth memory module in NU
    R64(3,1,k) = MR(sbase+OP(cp,11)+1,k); % memory for reverse network
    R64(1,1,k) = MM(sbsel,k);             % memory for network mapping data
    R64(13,1,k) = Mnull(sbsel,k);         % memory for null synapse flag
    R64(2,1,k) = MX(R64(1,2,k)+OP(cp,9)+1,k);  % get neuron outputs
    if (cPh==3)                % address of synapse (weight) selection
      R64(4,1,k) = R64(3,2,k);
    else
      R64(4,1,k) = Csb(2);
    end
  end
  
  % Synapse Unit
  
  for k=1:64    % loop(k) computes SNU(k)
    if R64(13,3,k)        % reset if null synapse (3)
      x_in = 0; 
    else
      x_in = R64(2,2,k); 
    end
    st = (x_in>=2);       % extract state
    real_y = x_in-st*2;
    if (cPh==3)           % (3)
      R64(6,1,k) = st;
    else
      R64(6,1,k) = real_y;
    end
    R64(5,1,k) = MW(R64(4,2,k)+OP(cp,10)+1,k);
    if (cPh~=1 && Csben(4) && ~Q64(3,mod(ck-2+200,200)+1,k))
      Ft(192+k,ck6) = 1; % multiplier
      slot_count(cPh) = slot_count(cPh) + 1;
      C64(1,1,k) = R64(5,2,k) * R64(6,2,k);     % compute weighted input (4)
    end
    Q64(1,mod(ck+130,200)+1,k) = R64(5,2,k);    % queue for weight (4)
    Q64(2,mod(ck+118,200)+1,k) = real_y;        % queue for x input (3)
    Q64(3,mod(ck-1+200,200)+1,k) = R64(13,3,k); % queue for null synapse flag (3)
    if ((nPh==2 || nPh==4) && Csben(121) && ~Q64(3,mod(ck+81,200)+1,k)) %(121)
      Ft(k,ck6) = 1; % mark a multiplication
      slot_count(nPh) = slot_count(nPh) + 1;
      C64(2,1,k) = Q64(2,mod(ck,200)+1,k) * R(18,2);  % product of neurons
    end
    if (nPh==2 && Csben(127))          % positive product saved in Mdw (127)
      Mdw(Csb(127)+1,k) = C64(2,7,k);
    end
    R64(9,1,k) = C64(2,7,k);           % negative product bypassed
    if (nPh==4&&Csben(127))            % read positive product (127)
      R64(10,1,k) = Mdw(Csb(127)+1,k);
    end
    if (nPh==4 && Csben(128) && ~Q64(3,mod(ck+74,200)+1,k)) % (128)
      Ft(64+k,ck6) = 3; % subtractor
      slot_count(nPh) = slot_count(nPh) + 1;
      C64(3,1,k) = R64(10,2,k) - R64(9,2,k); % epsilon*(posprods-negprods)
    end
    if (nPh==4 && Csben(134) && ~Q64(3,mod(ck+68,200)+1,k)) % (134)
      Ft(128+k,ck6) = 2; % mark an addition
      slot_count(nPh) = slot_count(nPh) + 1;
      C64(4,1,k) = Q64(1,mod(ck,200)+1,k) + C64(3,7,k); % update weights
    end
    if (nPh==4 && Csben(140))          % save (140)
      MW(Csb(140)+OP(ncp,10)+1,k) = C64(4,7,k);
      if (Csb(140)+1==6370)
        nop = 0;
      end;
    end
    R64(12,1,k) = C64(1,7,k);          % SNU output (10)
  end

  % verify
  if (nPh==4 && Csben(140))
    for k=1:64
      c_bunch_i = Csb(140);
      c_neuron_i = floor(c_bunch_i/OP(ncp,2));
      c_synapse_i = mod(c_bunch_i,OP(ncp,2))*64+k-1-c_neuron_i;
      if c_synapse_i<0
        c_synapse_i = c_synapse_i + OP(ncp,2) * 64;
      end
      if (c_synapse_i>=0 && c_synapse_i<OP(ncp-1,5)) && c_neuron_i>=0 && c_neuron_i<OP(ncp,5)
        t_1(c_synapse_i+1, c_neuron_i+1) = C64(4,7,k);
      end
    end
  end
  if (nPh==4 && Csben(128))
    for k=1:64
      c_bunch_i = Csb(128);
      c_neuron_i = floor(c_bunch_i/OP(ncp,2));
      c_synapse_i = mod(c_bunch_i,OP(ncp,2))*64+k-1-c_neuron_i;
      if c_synapse_i<0
        c_synapse_i = c_synapse_i + OP(ncp,2) * 64;
      end
      if (c_synapse_i>=0 && c_synapse_i<OP(ncp-1,5)) && c_neuron_i>=0 && c_neuron_i<OP(ncp,5)
        t_2(c_synapse_i+1, c_neuron_i+1) = R64(10,2,k); % pog
        t_3(c_synapse_i+1, c_neuron_i+1) = R64(9,2,k); % neg
      end
    end
  end

  % Dendrite Unit
  
  if (cPh~=1 && Csben(11))         % adder tree (11)
    n1 = Q64(3,mod(ck+191,200)+1,(1:32)*2  )==0;
    n2 = Q64(3,mod(ck+191,200)+1,(1:32)*2-1)==0;
    R64(15,1,32:63) = (n1 + n2)>0; % 1 if at least one of two is null
    Ft(256+(1:32),ck6) = 2 * ((n1 .* n2)>0); % mark if both are not null
    slot_count(cPh) = slot_count(cPh) + nnz((n1 .* n2)>0);
    C64(5,1,32:63) = R64(12,2,(1:32)*2-1) + R64(12,2,(1:32)*2);
  end
  if (cPh~=1 && Csben(17))         % (17)
    n1 = R64(15,7,(16:31)*2)==0; n2 = R64(15,7,(16:31)*2+1)==0;
    R64(15,1,16:31) = (n1 + n2) > 0;
    Ft(256+(33:48),ck6) = 2 * (( n1 .* n2)==0);
    slot_count(cPh) = slot_count(cPh) + nnz((n1 .* n2)==0);
    C64(5,1,16:31)  = C64(5,7,(16:31)*2) + C64(5,7,(16:31)*2+1);
  end
  if (cPh~=1 && Csben(23))        % (23)
    valid_com = (R64(15,7,(8:15)*2)==0) .* (R64(15,7,(8:15)*2+1)==0);
    R64(15,1,8:15) = (R64(15,7,(8:15)*2)>0) + (R64(15,7,(8:15)*2+1)>0);
    Ft(256+(49:56),ck6) = 2 * valid_com;
    slot_count(cPh) = slot_count(cPh) + nnz(valid_com);
    C64(5,1,8:15)  = C64(5,7,(8:15)*2) + C64(5,7,(8:15)*2+1);
  end
  if (cPh~=1 && Csben(29))        % (29)
    valid_com = (R64(15,7,(4:7)*2)==0) .* (R64(15,7,(4:7)*2+1)==0);
    R64(15,1,4:7) = (R64(15,7,(4:7)*2)>0) + (R64(15,7,(4:7)*2+1)>0);
    Ft(256+(57:60),ck6) = 2 * valid_com;
    slot_count(cPh) = slot_count(cPh) + nnz(valid_com);
    C64(5,1,4:7)  = C64(5,7,(4:7)*2) + C64(5,7,(4:7)*2+1);
  end
  if (cPh~=1 && Csben(35))        % (35)
    valid_com = (R64(15,7,(2:3)*2)==0) .* (R64(15,7,(2:3)*2+1)==0);
    R64(15,1,2:3) = (R64(15,7,(2:3)*2)>0) + (R64(15,7,(2:3)*2+1)>0);
    Ft(256+(61:62),ck6) = 2 * valid_com;
    slot_count(cPh) = slot_count(cPh) + nnz(valid_com);
    C64(5,1,2:3)  = C64(5,7,(2:3)*2) + C64(5,7,(2:3)*2+1);
  end
  if (cPh~=1 && Csben(41))        % (41)
    Ft(256+63,ck6) = 2;
    slot_count(cPh) = slot_count(cPh) + 1;
    C64(5,1,1)  = C64(5,7,2) + C64(5,7,3);
  end
  bpn = OP(cp,2);
  if (cPh~=1 && Csben(47))        % (47)
    Ft(320,ck6) = 4;              % mark accumulation
    slot_count(cPh) = slot_count(cPh) + 1;
    COM(1,1) = C64(5,7,1) + COM(1,2)*(mod(Csb(47),bpn)~=0);
  end
  nren = (Csben(52) && mod(Csb(52),bpn)==bpn-1); % neuron slot begins
  nrsel = nrsel + (nren && Csb(52)>bpn);
  if (nren) 
    net_sum = COM(1,6);           % timing depends on bpn
    Mnet(nrsel+1) = net_sum;
  end                             % latch netsum
  R(10,1) = net_sum;              % R(10,2): output of DU

  % Soma Unit
  
  % Sigmoid
  COM(2,1) = R(10,2) + R(4,2);        % sb(65) when bpn=13 nr(2)
  COM(3,1) = COM(2,7) * (-1);         % nr(8)
  COM(4,1) = exp(COM(3,7));           % nr(14) exp. ip []
  COM(5,1) = COM(4,7) + 1;            % nr(20)
  COM(6,1) = 1 / COM(5,7);            % nr(26) sigmoid result
  if nPh~=1 && Cnren(2)  Ft(321,ck6) = 2; slot_count(nPh) = slot_count(nPh) + 1; end        
  if nPh~=1 && Cnren(8) Ft(322,ck6) = 1; slot_count(nPh) = slot_count(nPh) + 1; end
  if nPh~=1 && Cnren(14) Ft(323,ck6) = 6; slot_count(nPh) = slot_count(nPh) + 1; end
  if nPh~=1 && Cnren(20) Ft(324,ck6) = 2; slot_count(nPh) = slot_count(nPh) + 1; end
  if nPh~=1 && Cnren(26) Ft(325,ck6) = 5; slot_count(nPh) = slot_count(nPh) + 1; end

  % SNU feedback
  COM(10,1) = epsilonw * COM(6,7);    % nr(32) apply learning rate
  R(17,1) = COM(10,7);                % send to all SNUs
  sw_sel = 33 - OP(ncp,2);            % variable length queue
  if nPh~=1 && Cnren(32) Ft(326,ck6) = 1; slot_count(nPh) = slot_count(nPh) + 1; end
  R(18,1) = R(17,sw_sel);             % (120)
  if (nPh==4 && Csben(121))           % for debugging
    Mdebug1(Csb(121)+1) = R(18,2);
  end

  % bias computation
  Boff = OP(ncp,7);
  R(4,1) = MB(nrsel+Boff+1);          % nr(1) read bias memory
  R(12,1) = R(4,2);                   % nr(2)
  if (nPh==2)                         % save Sigmoid output (probs) in MAp
    R(16,1) = COM(6,7);               % two write for one bias
    if Cnren(32) R(7,1)=1; R(6,1)=Cnr(32)+Boff; R(8,1) = COM(6,7);end
    if Cnren(33) R(7,1)=1; R(6,1)=Cnr(33)+OP(ncp,12); R(8,1) = R(16,2);end
  end
  In_n = OP(ncp,8);                   % shortcut
  if (nPh==3 || nPh==4)
    R(11,1) = COM(6,7);
    R(5,1) = MAp(Cnr(32)+Boff+1);     % nr(32)
    if Cnren(33)                      % nr(33)
      Ft(327,ck6) = 3;
      slot_count(nPh) = slot_count(nPh) + 1;
      COM(7,1) = R(5,2) - R(11,2);
    end
    if (nPh==3 && Cnren(33))          % nr(33)
      err = err + (R(5,2) - R(11,2))^2;
    end
    if (Cnren(39))                    % nr(39)
      Ft(328,ck6) = 1;
      slot_count(nPh) = slot_count(nPh) + 1;
      COM(13,1) = COM(7,7) * epsilonb;
    end
    if (Cnren(45))                    % nr(45)
      Ft(329,ck6) = 2;
      slot_count = slot_count + 1;
      COM(9,1) = R(12,44) + COM(13,7);
    end
    if (Cnren(51))                    % nr(51)
      MB(Cnr(51)+Boff+1) = COM(9,7);
    end
  end
  
  % apply random function
  R(13,1) = COM(6,7);       % nr(32)
  if (nPh==2)
    if (Cnren(32))          % nr(32)
      Ft(330,ck6) = 7;  % mark compare operation
      slot_count(nPh) = slot_count(nPh) + 1;
      if (ncp==2)       % this is for debug
        COM(12,1) = (myrand(Cnr(32)+1)<=COM(6,7));
      elseif (ncp==5)
        COM(12,1) = (myrand2(Cnr(32)+1)<=COM(6,7));
      elseif (ncp==8)
        COM(12,1) = (myrand3(Cnr(32)+1)<=COM(6,7));
      else
        COM(12,1) = (rand()<=COM(6,7));
      end
    end
    if (Cnren(38))          % nr(38)
      R(15,1) = COM(12,7) * 2 + R(13,7);
    end
  else
    R(15,1) = R(13,7);      % nr(38)
  end
  if (In_n==0 && nPh~=4)    % write in MX. nr(39)
    R(1,1) = Cnren(39);
    R(2,1) = Cnr(39);
    R(3,1) = R(15,2);
  end
  
  % training data input
  if In_n>0                 % train data input
    if dp_i<In_n&&~R(1,1) % save in activation memory and X memories
      c_i = mod(d_i,Nc)+1; b_i = floor(d_i/Nc)+1; % get train data indexes
      R(1,1) = 1; R(2,1) = dp_i; R(3,1) = batchdata(c_i,dp_i+1,b_i); % X
      R(7,1) = 1; R(6,1) = dp_i; R(8,1) = batchdata(c_i,dp_i+1,b_i); % A
      dp_i = dp_i + 1;         % adjust train data pointers
      if (nPh==1&&dp_i>=In_n) done = 1; end
    end
  end
  if (Cnr(39)==OP(ncp,5)-1 && cp==ncp)    % nr(39)
    done = 1;
  end
  % change phase
  if done
    if (nPh==3) fprintf('(mse = %6.1f)...', err); err = 0; end

    
    % for debugging
%%%%%%%%% LAYER 1 - 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if cp==4
      data = batchdata(:,:,1);
      org_err = 0;
      for i=1:Nn(1)
        posvisact(i)   = data(1,i);
      end
      for j=1:Nn(2)
        net1(j) = sum(data(1,:)' .* vishid_1(:,j));
        poshidprobs(1,j) = 1./(1 + exp(-net1(j) - hidbiases(j)));
        data2(j) = poshidprobs(1,j);
        posprods(:,j) = data(1,:)' * poshidprobs(1,j);
        poshidstates(1,j) = poshidprobs(1,j) > myrand(j);
      end
      for i=1:Nn(1)
        net2(i) = sum(poshidstates(1,:)*vishid_1(i,:)');
        negdata(i) = 1./(1 + exp(-net2(i) - visbiases(i)));
        negvisact(i) = negdata(i);
        visbiasinc(i) = epsilonb*(posvisact(i)-negvisact(i));
        visbiases(i) = visbiases(i) + visbiasinc(i);

        org_err = org_err + (data(1,i)-negdata(i)).^2;
      end
      for j=1:Nn(2)
        net3(j) = sum(negdata' .* vishid_1(:,j));
        neghidprobs(1,j) = 1./(1 + exp(-net3(j) - hidbiases(j)));
        negprods(:,j) = negdata' * neghidprobs(1,j);
        hidbiasinc(j) = epsilonb*(poshidprobs(1,j)-neghidprobs(1,j));
        hidbiases(j) = hidbiases(j) + hidbiasinc(j);

        vishidinc(:,j) =  epsilonw*posprods(:,j)-epsilonw*negprods(:,j);
        vishid_1(:,j) = vishid_1(:,j) + vishidinc(:,j);
      end
      fprintf(1, 'mse %6.1f  ', org_err); 
        myrand = rand(1,Nn(2));
    end
%%%%%%%%% LAYER 2 - 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if cp==7
      org_err = 0;
      for i=1:Nn(2)
        posvisact2(i)   = data2(i);
      end
      for j=1:Nn(3)
        net1(j) = sum(data2' .* vishid_2(:,j));
        poshidprobs2(1,j) = 1./(1 + exp(-net1(j) - hidbiases2(j)));
        data3(j) = poshidprobs2(1,j);
        posprods2(:,j) = data2' * poshidprobs2(1,j);
        poshidstates2(1,j) = poshidprobs2(1,j) > myrand2(j);
      end
      for i=1:Nn(2)
        net2(i) = sum(poshidstates2(1,:)*vishid_2(i,:)');
        negdata2(i) = 1./(1 + exp(-net2(i) - visbiases2(i)));
        negvisact2(i) = negdata2(i);
        visbiasinc2(i) = epsilonb*(posvisact2(i)-negvisact2(i));
        visbiases2(i) = visbiases2(i) + visbiasinc2(i);

        org_err = org_err + (data2(i)-negdata2(i)).^2;
      end
      for j=1:Nn(3)
        net3(j) = sum(negdata2(1,:)' .* vishid_2(:,j));
        neghidprobs2(1,j) = 1./(1 + exp(-net3(j) - hidbiases2(j)));
        negprods2(:,j) = negdata2' * neghidprobs2(1,j);
        hidbiasinc2(j) = epsilonb*(poshidprobs2(1,j)-neghidprobs2(1,j));
        hidbiases2(j) = hidbiases2(j) + hidbiasinc2(j);

        vishidinc2(:,j) =  epsilonw*posprods2(:,j) - epsilonw*negprods2(:,j);
        vishid_2(:,j) = vishid_2(:,j) + vishidinc2(:,j);
      end
      fprintf(1, '********** mse %6.1f  ', org_err); 
      myrand2 = rand(1,Nn(3));
    end
%%%%%%%%% LAYER 3 - 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if cp==10
      org_err = 0;
      for i=1:Nn(3)
        posvisact3(i)   = data3(i);
      end
      for j=1:Nn(4)
        net1(j) = sum(data3(1,:)' .* vishid_3(:,j));
        poshidprobs3(1,j) = 1./(1 + exp(-net1(j) - hidbiases3(j)));
        data4(j) = poshidprobs3(1,j);
        posprods3(:,j) = data3' * poshidprobs3(1,j);
        poshidstates3(1,j) = poshidprobs3(1,j) > myrand3(j);
      end
      for i=1:Nn(3)
        net2(i) = sum(poshidstates3(1,:)*vishid_3(i,:)');
        negdata3(i) = 1./(1 + exp(-net2(i) - visbiases3(i)));
        negvisact3(i) = negdata3(i);
        visbiasinc3(i) = epsilonb*(posvisact3(i)-negvisact3(i));
        visbiases3(i) = visbiases3(i) + visbiasinc3(i);

        org_err = org_err + (data3(i)-negdata3(i)).^2;
      end
      for j=1:Nn(4)
        net3(j) = sum(negdata3(1,:)' .* vishid_3(:,j));
        neghidprobs3(1,j) = 1./(1 + exp(-net3(j) - hidbiases3(j)));
        negprods3(:,j) = negdata3' * neghidprobs3(1,j);
        hidbiasinc3(j) = epsilonb*(poshidprobs3(1,j)-neghidprobs3(1,j));
        hidbiases3(j) = hidbiases3(j) + hidbiasinc3(j);

        vishidinc3(:,j) =  epsilonw*posprods3(:,j)-epsilonw*negprods3(:,j);
        vishid_3(:,j) = vishid_3(:,j) + vishidinc3(:,j);
      end
      fprintf(1, '******************** mse %6.1f  ', org_err); 
      myrand3 = rand(1,Nn(4));
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     if (cp==5)   % after L1-L2 stage
%       fprintf('Layer L1 - L2\n');
%       diff = t_1(1:784,1:500) - vishid_1(1:784,1:500);
%       df = find(abs(diff)>0.0001);
%       if size(df>0) 
%         fprintf('Number of mismatch in weight updates = %d\n', size(df));
%       end
%       df = find(abs(MB(1:784) - visbiases(1:784))>0.0001)
%       if size(df>0) 
%         fprintf('Number of mismatch in bias of vis = %d\n', size(df));
%       end
%       df = find(abs(MB(2001:2500) - hidbiases(1:500))>0.0001)
%       if size(df>0) 
%         fprintf('Number of mismatch in bias of hid = %d\n', size(df));
%       end
%     end
%     if (ncp==8)   % after L2-L3 stage
%       fprintf('Layer L2 - L3\n');
%       df = find(abs(tempweight1(1:500,1:500) - vishid_2(1:500,1:500))>0.0001)
%       if size(df>0) 
%         [df]
%         fprintf('Number of mismatch in weight updates = %d\n', df);
%       end
%       df = find(abs(MB(1:784) - visbiases(1:784))>0.0001)
%       if size(df>0) 
%         [df]
%         fprintf('Number of mismatch in bias of vis = %d\n', df);
%       end
%       df = find(abs(MB(2001:2500) - hidbiases(1:500))>0.0001)
%       if size(df>0) 
%         [df]
%         fprintf('Number of mismatch in bias of hid = %d\n', df);
%       end
%     end

    if (nPh==4)
      fprintf('\nclock=%6d\tslot=%6d,%6d,%6d\n', ck, slot_count(2), slot_count(3), slot_count(4));
      slot_count = zeros(1,10);
    end
    cp = cp + 1;
    if (OP(cp,1)==9)      % goto statement
      cp = OP(cp,2);
    end;
    cp_cnt = 0;           % for defer phase change for tail part of pipeline
    fprintf('%7d phase %2d...', ck, cp);
    if (OP(cp,8)>0)       % reset train-data pointer
      dp_i = 0;
    end
    % reset variables
    sbase = 0; s_en = 1; nrsel = 0; nren = 0; done = 0;
  else
    if (sbase+1>=OP(cp,3)||cPh==1)
      s_en = 0;
    else
      sbase = sbase + 1;
    end
  end
  cp_cnt = cp_cnt + 1;
  if (cp_cnt==50)
    ncp = cp;
  end
  if (d_i>Nc*Nb)          % all done
    break;
  end
  % write on X(in NU) and Act(in SU) memories when request pending
  if (R(7,2)) 
    MAp(R(6,2)+1) = R(8,2); 
  end
  if (R(1,2)) 
    MX(R(2,2)+OP(ncp,6)+1,1:64) = R(3,2); 
  end
  
  % Control Unit
  
  %  shifting registers on clock tick
  Csb(2:150)=Csb(1:149);Csben(2:150)=Csben(1:149);Csb(1)=sbase;Csben(1)=s_en;
  Cnr(2:150)=Cnr(1:149);Cnren(2:150)=Cnren(1:149);Cnr(1)=nrsel;Cnren(1)=nren;
  R([12,13,17],6:60)=R([12,13,17],5:59);R(:,2:5)=R(:,1:4);R(:,1)=0;
  R64(:,2:7,:)=R64(:,1:6,:);R64(:,1,:)=0;
  COM(:,2:10)=COM(:,1:9);COM(:,1)=0;C64(:,2:10,:)=C64(:,1:9,:);C64(:,1,:)=0;
  if mod(ck,200*DspRes)==0
    if mod(ck,400*DspRes)==0 
      view = [view(1:330,200*DspRes+1:1000*DspRes), Ft(1:330,200*DspRes+1:400*DspRes)];
      Ft(1:330,200*DspRes+1:400*DspRes) = 0;
    else
      view = [view(1:330,200*DspRes+1:1000*DspRes), Ft(1:330,1:200*DspRes)];
      Ft(1:330,1:200*DspRes) = 0;
    end
    set(0,'CurrentFigure',fgr), colormap(cmap), image(view(1:330,1:1000*DspRes)+1), caxis([0, 9]), colorbar('YTickLabel',{'idle','multiplier','adder','subtracter','accumulator','reciprocal','exponent','compare'}), axis xy;
    drawnow();
  end
end
