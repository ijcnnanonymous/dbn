% Hardware Simulator for
% Ristricted Boltzmann Machine in Neuron Machine Hardware Architecture
% Version 1.000
% Code provided by IJCNN_Anonymous (to be changed), Jan.2014
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
% 
clear all;clf;
I=imadjust(imresize(rgb2gray(imread('blason.jpg')),1.15),[0 0.8],[]);
I=1-(double(I)/255); I3=I(51:78,71:98+28*7);
for i=0:7 DB(i+1,:,1)=reshape(I3(:,(1:28)+28*i-i+(i>4)*4),1,[]);end;
% converter; makebatches; DB=batchdata; % for MNIST. Needs .m files
Nn=[784 500 500 2000];[Nc Nn(1) Nb]=size(DB);
ew=0.1;eb=0.1;MW=zeros(6E4,64);MD=zeros(1E4,64);MM=MW;MR=MW;
MN=MW+1;MX=zeros(1E4,64);MB=zeros(1,2E4);MA=MB; SOT=zeros(11,12);
moff=0;woff=0;roff=0;SOT(1,[1,5])=[1,Nn(1)];oi=2;
for l=1:3   % set up block memories and stage operation table
  hb=ceil(Nn(l)/64);hNb=hb*Nn(l+1);vb=ceil(Nn(l+1)/64);vNb=vb*Nn(l);
  for j=0:Nn(l+1)-1 for i=0:Nn(l)-1
    hinx=mod(i+j,hb*64);hbi=floor(hinx/64);snu_i=mod(hinx,64);
    ma=j*hb+hbi+moff+1;wa=j*hb+hbi+woff+1;
    MW(wa,snu_i+1)=0.1*randn();MM(ma,snu_i+1)=i;MN(ma,snu_i+1)=0;
    vinx=mod(i+j,vb*64);vbi=floor(vinx/64);
    ma=i*vb+vbi+hNb+moff+1;ra=i*vb+vbi+roff+1;
    MR(ra,snu_i+1)=j*hb+hbi;MM(ma,snu_i+1)=j;MN(ma,snu_i+1)=0;
  end; end;
  k2=2E3;k4=4E3;of1=mod(l,2)*k2; of2=mod(l+1,2)*k2; bo=(l-1)*k4;
  SOT(oi+0,:)=[2,Nn(l+1),hb,hNb,0,moff,of2,of1,0,woff,bo+k2,l*k4];
  SOT(oi+1,:)=[3,Nn(l),vb,vNb,0,moff+hNb,of1,k4,roff,woff,bo,0];
  SOT(oi+2,:)=[4,Nn(l+1),hb,hNb,0,moff,k4,0,0,woff,bo+k2,0];
  oi=oi+3; moff=moff+hNb+vNb; woff=woff+hNb; roff=roff+vNb;
end
SOT(oi-1,5)=Nn(1);SOT(oi,1:2)=[9,2];Ct=uint16(zeros(1,300));[SOT]
Csb=Ct;Csbe=Ct;Cnr=Ct;Cnre=Ct;Q=zeros(10,200,64);R=zeros(20,60);
RA=zeros(20,130,64);C=R;CA=RA;dcnt=0;dpi=0;di=0;B=DB(1,:,1);
net_sum=0;x_in=zeros(1,64);sb=0;sbe=0;nr=0;nre=0;eos=0;er=0;
cp=1;np=1;cS=SOT(1,:);nS=SOT(1,:);hold all;
% starting machine
for ck=1:1E9     % clock cycles
  hS=cS(1);tS=nS(1);bpn=cS(3);bf=nS(11);mf=sb+cS(6)+1;  % shortcuts
  for k=1:64  % each k represents k'th memory module and SNU
    % Network Unit
    RA(3,1,k)=MR(sb+cS(9)+1,k);RA(1,1,k)=MM(mf,k);RA(13,1,k)=MN(mf,k);
    RA(2,1,k)=MX(RA(1,2,k)+cS(7)+1,k);
    if hS==3 RA(4,1,k)=RA(3,2,k);else RA(4,1,k)=Csb(2);end;
    % Synapse Unit
    x_in=RA(2,2,k);x_in(RA(13,3,k)>0)=0;st=(x_in>=2);r_y=x_in-st*2;
    if hS==3 RA(6,1,k)=st;else RA(6,1,k)=r_y;end;
    RA(5,1,k)=MW(RA(4,2,k)+cS(10)+1,k);
    CA(1,1,k)=RA(5,2,k).*RA(6,2,k);RA(12,1,k)=CA(1,7,k);
    Q(1,mod(ck+130,200)+1,k)=RA(5,2,k);Q(2,mod(ck+118,200)+1,k)=r_y;
    CA(2,1,k)=Q(2,mod(ck,200)+1,k).*R(18,2);RA(9,1,k)=CA(2,7,k);
    if tS==2&&Csbe(127) MD(Csb(127)+1,k)=CA(2,7,k);end;
    RA(10,1,k)=MD(Csb(127)+1,k);CA(3,1,k)=RA(10,2,k)-RA(9,2,k);
    CA(4,1,k)=Q(1,mod(ck,200)+1,k)+CA(3,7,k);
    if tS==4&&Csbe(140) MW(Csb(140)+nS(10)+1,k)=CA(4,7,k);end;
  end
  % Dendrite Unit
  CA(5,1,32:63)=RA(12,2,(1:32)*2-1)+RA(12,2,(1:32)*2); % adder tree
  CA(5,1,1:31)=CA(5,7,(1:31)*2)+CA(5,7,(1:31)*2+1);
  C(1,1)=CA(5,7,1)+C(1,2)*(mod(Csb(47),bpn)~=0);
  nre=(Csbe(52)&&mod(Csb(52),bpn)==bpn-1);nr=nr+(nre&&Csb(52)>bpn);
  if nre net_sum=C(1,6);end;R(10,1)=net_sum;
  % Soma Unit
  C(2,1)=R(10,2)+R(4,2);C(3,1)=-1*C(2,7);C(4,1)=exp(C(3,7));
  C(5,1)=C(4,7)+1;C(6,1)=1/C(5,7);C(10,1)=ew*C(6,7);
  R(17,1)=C(10,7);R(18,1)=R(17,33-nS(3));R(13,1)=C(6,7);
  R(4,1)=MB(nr+bf+1);R(12,1)=R(4,2);In_n=nS(5);
  if tS==2
    R(16,1)=C(6,7);if Cnre(32)||Cnre(33) R(7,1)=1;end;
    if Cnre(32) R(6,1)=Cnr(32)+bf;R(8,1)=C(6,7);end;
    if Cnre(33) R(6,1)=Cnr(33)+nS(12);R(8,1)=R(16,2);end;
    R(14,1)=floor(rand()*2^24);C(11,1)=R(14,2)/2^24;
    C(12,1)=C(11,7)<=C(6,7);if Cnre(38) R(15,1)=C(12,7)*2+R(13,7);end;
  elseif tS==3 || tS==4
    R(11,1)=C(6,7);R(5,1)=MA(Cnr(32)+bf+1);C(7,1)=R(5,2)-R(11,2);
    if tS==3&&Cnre(33) er=er+(R(5,2)-R(11,2))^2;end;
    C(13,1)=C(7,7)*eb;C(9,1)=R(12,44)+C(13,7);
    if Cnre(51) MB(Cnr(51)+bf+1)=C(9,7);end;
  end
  if tS~=2 R(15,1)=R(13,7);end;
  if In_n==0&&tS~=4 R(1,1)=Cnre(39);R(2,1)=Cnr(39);R(3,1)=R(15,2);end;
  if In_n>0&&dpi<In_n&&~R(1,1) % inject train data input
    if dpi+1>=In_n if tS==1 eos=1;end;I=reshape(MA(1:784),28,28);
      figure(1),subplot(2,2,1),imshow(I),title('V1-pos');drawnow();end;
    dpi=dpi+1;R(1,1)=1;
    R(2,1)=dpi-1;R(3,1)=B(dpi);R(7,1)=1;R(6,1)=dpi-1;R(8,1)=B(dpi);
  end
  if Cnr(39)==nS(2)-1&&cp==np eos=1;end;
  if eos  % Control Unit
    if np==3 I=reshape(MX((1:784)+nS(8),1),28,28);
      subplot(2,2,2),imshow(I),title('V1-neg');drawnow();end;
    if np>4&&hS==2 I=reshape(MX((1:nS(2))+nS(8)),25,floor(nS(2)/25));
      subplot(2,2,floor((np-2)/3)+2),imshow(I),title('H2-H3');drawnow();end;
    if hS==4 fprintf('(Err=%6.1f)\t',er);er=0;end; cp=cp+1;cS=SOT(cp,:);
    if cS(1)==9 cp=cS(2);cS=SOT(cp,:);disp(di);end;
    if cS(5)>0 if di>=Nc*Nb di=1;else di=di+1;end; dpi=0;
      B=DB(mod(di,Nc)+1,:,floor((di-1)/Nc)+1);end;
    fprintf('%2d ',cp);sb=0;sbe=1;nr=0;nre=0;eos=0;dcnt=0;
  else if sb+1>=cS(4)||hS==1 sbe=0;else sb=sb+1;end;end;
  dcnt=dcnt+1;if dcnt==50 np=cp;nS=cS;end;
  if R(7,2) MA(R(6,2)+1)=R(8,2);end;lr=[12,13,17];
  if R(1,2) MX(R(2,2)+nS(8)+1,:)=R(3,2);end;t=2:150;f=1:149;
  Csb(t)=Csb(f);Csbe(t)=Csbe(f);Csb(1)=sb;Csbe(1)=sbe;Cnr(t)=Cnr(f);
  Cnre(t)=Cnre(f);Cnr(1)=nr;Cnre(1)=nre;R(lr,6:60)=R(lr,5:59);
  R(:,2:7)=R(:,1:6);R(:,1)=0;RA(:,2:7,:)=RA(:,1:6,:);RA(:,1,:)=0;
  C(:,2:7)=C(:,1:6);C(:,1)=0;CA(:,2:7,:)=CA(:,1:6,:);CA(:,1,:)=0;
end
