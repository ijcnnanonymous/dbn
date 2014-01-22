clear all;clf;

% Use characters from an image stored in MATLAB as training data.
%   characters: A,Z,E,R,G,U,R,E,S

I=imadjust(imresize(rgb2gray(imread('blason.jpg')),1.15),[0 0.8],[]);
I=1-(double(I)/255);    % convert to real format
I3=I(51:78,71:98+28*7); % saparate digites
for i = 0:7
  DB(i+1,:,1) = reshape(I3(:,(1:28)+28*i-i+(i>4)*4),1,[]);
end;

% This code was originally designed for MNIST handwritten database.
% To change the training data, disable lines 6-11 above,
%   and enable lines 19-21.
% In this case, you need converter.m, makebatches.m, and MNIST files
%   from http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html.

% converter;    % convert MNIST tiles to MATLAB form
% makebatches;  % build batch
% DB = batchdata;

Nn=[784 500 500 2000];    % define network size
[Nc Nn(1) Nb] = size(DB); % read database size
ew = 0.1; eb = 0.1;       % learning rates

% Define block memories

MW = zeros(6E4,64);   % wij
MD = zeros(1E4,64);   % temporary for (vi * hj)
MM = MW;              % network topology
MR = MW;              % reverse network topology
MN = MW + 1;          % flag for null connections (initially all set)
MX = zeros(1E4,64);   % duplicate neuron states
MB = zeros(1,2E4);    % neuron bias
MA = MB;              % temporary for neuron bias

% Initialize block memories and Stage operation Table

SOT = zeros(11,12);
moff = 0; woff = 0; roff = 0; % memory offsets
SOT(1,[1,5]) = [1,Nn(1)];     % set the first record to a type 1 stage
oi = 2;                       % current point of SOT

for l = 1:3
  hb = ceil(Nn(l) / 64);      % bpn for hidden layer
  hNb = hb * Nn(l + 1);       % total number of SBs for hidden layer
  vb = ceil(Nn(l + 1) / 64);  % bpn for visible layer
  vNb = vb * Nn(l);           % total number of SBs for visible layer
  
  for j = 0:Nn(l+1) - 1
    for i = 0:Nn(l) - 1
      % set for forward network
      hinx = mod(i + j, hb * 64); % redefine the position of fwd. conn.
      hbi = floor(hinx / 64);     % index of SB for the new position
      snu_i = mod(hinx, 64);      % index of SNU for the new position
      ma = j * hb + hbi + moff + 1; % current address of MM memory
      wa = j * hb + hbi + woff + 1; % current address of MW memory
      MW(wa, snu_i + 1) = 0.1 * randn();  % set wij of the connection
      MM(ma, snu_i + 1) = i;              % set mij of the connection
      MN(ma,snu_i+1) = 0;                 % unset null flag
      % set for reverse network
      vinx = mod(i + j, vb * 64); % redefine the position of bwd. conn.
      vbi = floor(vinx / 64);     % index of SB for the new position
      ma= i * vb + vbi + hNb + moff + 1;  % address of MR memory
      ra = i * vb + vbi + roff + 1;       % address of MM memory
      MR(ra, snu_i + 1) = j * hb + hbi;   % set pointer of wij
      MM(ma, snu_i + 1) = j;              % set mij of the connection
      MN(ma, snu_i + 1) = 0;              % unset null flag
    end;
  end;
  
  k2 = 2E3; k4 = 4E3;         % shortcuts
  of1 = mod(l, 2) * k2;       % destination MX block
  of2 = mod(l + 1, 2) * k2;   % source MX block
  bo = (l - 1) * k4;          % offset for bias memory
  
  % Stage Operation Table
  % The records are organized as:
    % SOT(:, 1): stage type
    % SOT(:, 2): number of neurons
    % SOT(:, 3): number of SBs for each neuron (bpn)
    % SOT(:, 4): total number of SBs
    % SOT(:, 5): number of training data (if any)
    % SOT(:, 6): address offset of MM
    % SOT(:, 7): address offset of read port of MX
    % SOT(:, 8): address offset of write port of MX
    % SOT(:, 9): address offset of MR
    % SOT(:,10): address offset of MW
    % SOT(:,11): address offset of MB
    % SOT(:,12): address offset of MA
  
  % Add three records in the SOT
  SOT(oi + 0, :) = [2,Nn(l+1),hb,hNb,0,moff,of2,of1,0,woff,bo+k2,l*k4];
  SOT(oi + 1, :) = [3,Nn(l),vb,vNb,0,moff+hNb,of1,k4,roff,woff,bo,0];
  SOT(oi + 2, :) = [4,Nn(l+1),hb,hNb,0,moff,k4,0,0,woff,bo+k2,0];
  oi = oi + 3;
  
  % Adjust memory offsets
  moff = moff + hNb + vNb;
  woff = woff + hNb;
  roff = roff + vNb;
end

SOT(oi - 1, 5) = Nn(1); % new training data is loaded at last stage
SOT(oi, 1:2) = [9, 2];  % set last record to "GO_TO 2" statement

Ct = uint16(zeros(1, 300)); % type definition (for shortcut)
[SOT]

% define shift registers for control data
Csb = Ct;   % the flow of SB indexes (from NU to DU)
Csbe = Ct;  % the flow of enable bits of SB (from NU to DU)
Cnr = Ct;   % the flow of neuron indexes (SU)
Cnre = Ct;  % the flow of enable bits of neuron (SU)
Q = zeros(10,200,64);   % FIFO queue
R = zeros(20, 60); RA = zeros(20,130,64);   % shist registers
C = R; CA = RA;                             % pipelined operator
dcnt = 0; % counter for delaying the stage change of tail part
dpi = 0;  % training data pointer (current neuron)
di = 0;   % training data pointer (current set)
B = DB(1,:,1);  % current SOT record = recore 1
sb = 0;   % register for SB index
sbe = 0;  % register for SB enable
nr = 0;   % register for neuron index
nre = 0;  % register for neuron enable
er = 0;   % sequre error accumulation
cp = 1;   % SOT pointer for head part (NU, SNU, and DU)
np = 1;   % SOT pointer for tail part (SU)
cS = SOT(1,:);  % SOT record for head part
nS = SOT(1,:);  % SOT record for tail part
net_sum = 0; x_in = zeros(1,64); eos = 0; % temporary
hold all;

% s t a r t i n g   m a c h i n e

for ck = 1:1E9      % system clock cycles
  
  % define shortcuts
  hS = cS(1);       % stage type for head part of system
  tS = nS(1);       % stage type for tail part of system
  bpn = cS(3);      % set bpn
  bf = nS(11);      % offset for MB
  mf = sb + cS(6) + 1;  % the address of MM and MN
  
  for k=1:64  % each k represents k'th memory module and SNU
    
    % Network Unit
    
    % input: MX write port (from SU)
    % output
      % RA(2,2,1:64)    : ymij output
      % RA(4,2,1:64)    : address of MW
      % RA(13,3,1:64)   : null connector indicator
      
    RA(3,1,k) = MR(sb + cS(9) + 1, k);  % read MR
    RA(1,1,k) = MM(mf,k);               % read MM
    RA(13,1,k) = MN(mf,k);              % read MN
    RA(2,1,k) = MX(RA(1,2,k) + cS(7) + 1, k); % read MX
    if hS==3   % MR output is selected when stage type = 3
      RA(4,1,k) = RA(3,2,k);
    else
      RA(4,1,k) = Csb(2);
    end;
    
    % Synapse Unit

    % input
      % RA(2,2,1:64)    : ymij output (from NU)
      % RA(4,2,1:64)    : address of MW (from NU)
      % RA(13,3,1:64)   : null connector indicator (from NU)
      % R(18,2)         : e * prob (from SU)
    % output: RA(12,2,1:64) : weighted input
    
    x_in = RA(2,2,k);
    x_in(RA(13,3,k)>0) = 0; % set to zero if it is null connection
    st = (x_in>=2);         % separate binary and 
    r_y = x_in - st * 2;    % real states
    if hS==3                % binary is used only in stage type 3
      RA(6,1,k) = st;       %   according to Hinton's code
    else
      RA(6,1,k) = r_y;
    end;
    RA(5,1,k) = MW(RA(4,2,k) + cS(10) + 1, k);  % read MW
    CA(1,1,k) = RA(5,2,k) .* RA(6,2,k); % computed weighted input
    RA(12,1,k) = CA(1,7,k);             % set the output register

    Q(1, mod(ck + 130, 200) + 1, k) = RA(5,2,k);      % queue weight
    Q(2, mod(ck + 118, 200) + 1, k)=r_y;              % real state
    CA(2,1,k) = Q(2, mod(ck, 200) + 1, k) .* R(18,2); % prod = vi * hj
    RA(9,1,k) = CA(2,7,k);
    if tS==2 && Csbe(127)               % MD <-- prod in stage type 2
      MD(Csb(127) + 1, k) = CA(2,7,k);
    end;
    RA(10,1,k) = MD(Csb(127) + 1, k);   % read MD
    CA(3,1,k) = RA(10,2,k) - RA(9,2,k); % compute prod_pos - prod_neg
    CA(4,1,k) = Q(1,mod(ck, 200) + 1, k) + CA(3,7,k); % add to weight
    if tS==4 && Csbe(140)               % save in MW in stage type 4
      MW(Csb(140) + nS(10) + 1, k) = CA(4,7,k);
    end;
  end
  
  % Dendrite Unit
  
  % input: RA(12,2,1:64) : weighted inputs from SNU
  % output: R(10,2) : sum of weighted input
  
  % adder tree with P - 1 adders
  CA(5,1,32:63) = RA(12,2,(1:32) * 2-1) + RA(12,2,(1:32) * 2);
  CA(5,1,1:31) = CA(5,7,(1:31) * 2) + CA(5,7,(1:31) * 2 + 1);
  % accumulator at the end of the tree
  C(1,1) = CA(5,7,1) + C(1,2) * (mod(Csb(47),bpn)~=0);
  % neuron index and enable are generated here
  % note: relative timing of neuron index changes depending to bpn
  nre = (Csbe(52) && mod(Csb(52),bpn)==bpn-1);
  nr = nr + (nre&&Csb(52)>bpn);
  if nre              % implement a latch
    net_sum = C(1,6);
  end;
  R(10,1)=net_sum;
  
  % Soma Unit
  
  % input
    % R(10,2) :   sum of weighted input (from DU)
    % B       :   training data (external) one neuron at a time
  % output
    % R(18,2) : e * prob
    % R(3,2)  : new neuron output

  % implement sigmoid function 1/(1+e(-x)) and compute prob
  C(2,1) = R(10,2) + R(4,2);
  C(3,1) = -1 * C(2,7);
  C(4,1) = exp(C(3,7));
  C(5,1) = C(4,7) + 1;
  C(6,1) = 1 / C(5,7);
  R(11,1) = C(6,7);         % the output of sigmoid function

  C(10,1) = ew * C(6,7);    % prod = e * prob
  R(17,1) = C(10,7);        % put prod in queue 
  R(18,1) = R(17, 33 - nS(3)); %   to synchronize with SNU
  
  R(4,1) = MB(nr + bf + 1); % read bias from MB
  R(12,1) = R(4,2);         % feed back for update bias
  In_n = nS(5);             % set the number of training data.
  if tS==2                  % in stage type 2
    % bias computation part
    % R(6,1), R(7,1), R(8,1) serve as a gateway to MA
    if Cnre(32) || Cnre(33) % prob is stored in two places of MA:
      R(7,1) = 1;      % for hpos, vpos of current and next layer
    end;               
    if Cnre(32)        % in the first slot store with offset of MB
      R(6,1) = Cnr(32) + bf;
      R(8,1) = C(6,7);
    end;
    if Cnre(33)        % in the second slot store with offset of MA
      R(6,1) = Cnr(33) + nS(12);
      R(8,1) = R(11,2);
    end;
    % random number generator
    R(14,1) = floor(rand() * 2^24); % simulate 24bit LFSR
    C(11,1) = R(14,2) / 2^24;
    C(12,1) = C(11,7)<=C(6,7); % compute binary state
    if Cnre(38)          % pack binary state and real data
      R(15,1) = C(12,7) * 2 + R(11,7);  % set new neuron state
    end;
  elseif tS==3 || tS==4  % compute biases in stage types 3 and 4
    R(5,1) = MA(Cnr(32) + bf + 1);  % read MA
    C(7,1) = R(5,2) - R(11,2);      % state difference
    if tS==3 && Cnre(33)            % accumulate square error
      er = er + (R(5,2) - R(11,2))^2;
    end;
    C(13,1) = C(7,7) * eb;        % apply learning rate
    C(9,1) = R(12,44) + C(13,7);  % add to previous biahn
    if Cnre(51) 
      MB(Cnr(51)+bf+1) = C(9,7);  % save in MB
    end;
    R(15,1) = R(11,7);            % set new neuron state
  end
  if In_n==0 && tS~=4   % write new state on MX if not input mode
    R(1,1) = Cnre(39);
    R(2,1) = Cnr(39);
    R(3,1) = R(15,2);
  end;

  % read training data
  if In_n>0 && dpi<In_n && ~R(1,1)  % inject train data input
    if dpi+1>=In_n                  % if all inputs are read
      if tS==1                      % end stage if stage type 1
        eos = 1;
      end;
      I = reshape(MA(1:784),28,28); % display new training data;
      figure(1), subplot(2,2,1), imshow(I), title('V1-pos');
      drawnow();
    end;
    dpi = dpi + 1;                  % point to input
    R(1,1) = 1;                     % write data in MX
    R(2,1) = dpi - 1;
    R(3,1) = B(dpi);
    R(7,1) = 1;                     % write data in MA
    R(6,1) = dpi - 1;
    R(8,1) = B(dpi);
  end
  
  if Cnr(39)==nS(2)-1 && cp==np     % end stage if all neurons done
    eos = 1;
  end;

  % Control Unit
  
  if eos  % end stage and set for the next stage
    if np==3  % draw image from first layer negative visible
      I = reshape(MX((1:784)+nS(8),1),28,28);   % get image
      subplot(2,2,2), imshow(I), title('V1-neg');
      drawnow();
    end;
    if np>4 && hS==2 % draw image from layer 2 and 3 of hidden
      I = reshape(MX((1:nS(2))+nS(8)),25,floor(nS(2)/25));
      subplot(2,2,floor((np-2)/3)+2),imshow(I),title('H2-H3');
      drawnow();
    end;
    if hS==4        % print error
      fprintf('(Err=%6.1f)\t',er);
      er = 0;
    end;
    cp = cp + 1;    % change to next stage (tail part remains unchanged)
    cS = SOT(cp,:); % set current record of SOT
    if cS(1)==9     % handle goto statement
      cp = cS(2);
      cS = SOT(cp,:);
      disp(di);     % print training sample number at the end of line
    end;
    if cS(5)>0      % if new stage has training data to read
      if di>=Nc * Nb    % advance pointer to the training data
        di = 1;
      else
        di = di + 1;
      end;
      dpi = 0;
      B = DB(mod(di,Nc)+1,:,floor((di-1)/Nc)+1); % shortcut
    end;
    fprintf('%2d ',cp);
    sb = 0;     % reset control registers
    sbe = 1;
    nr = 0;
    nre = 0;
    eos = 0;
    dcnt = 0;   % start tail-change delay
  % if not end of stage
  else
    if sb + 1>=cS(4) || hS==1 % reset enable if all SB index processed
      sbe = 0;
    else
      sb = sb + 1;
    end;
  end;
  dcnt = dcnt + 1;    % run tail-change delay counter
  if dcnt==50         % if counter is reached, set tail = head 
    np = cp;
    nS = cS;
  end;
  if R(7,2)           % gateway to MA memory
    MA(R(6,2)+1) = R(8,2);
  end;
  lr=[12,13,17];      % long registers
  if R(1,2)           % gateway to write ports of MX memory
    MX(R(2,2)+nS(8)+1,:) = R(3,2);
  end;
  t = 2:150; f = 1:149;   % shortcuts
  % shift control registers
  Csb(t) = Csb(f);
  Csbe(t) = Csbe(f);
  Csb(1) = sb;
  Csbe(1) = sbe;
  Cnr(t) = Cnr(f);
  Cnre(t) = Cnre(f);
  Cnr(1) = nr;
  Cnre(1) = nre;
  % shift registers
  R(lr,6:60) = R(lr,5:59);
  R(:,2:7) = R(:,1:6);
  R(:,1) = 0;
  RA(:,2:7,:) = RA(:,1:6,:);
  RA(:,1,:) = 0;
  % shift pipelined arithmetic operators
  C(:,2:7) = C(:,1:6);
  C(:,1) = 0;
  CA(:,2:7,:) = CA(:,1:6,:);
  CA(:,1,:) = 0;
end
