function [ best ] = bestexemplarhelper( mm,nn,m,n,img,Ip,toFill,sourceRegion )

% Initial values
patchErr=0.0; err=0.0; bestErr=1000000000.0; flag=0;
mn=m*n; mmnn=mm*nn;

% For each patch
N=nn-n+1;  M=mm-m+1;
for j = 1:N
   J = j+n-1;
   for i = 1:M
       I = i+m-1;
       
       % Calculate patch errors
       % for each pixel in the current patch
       jj2 = 1;
       for jj = j:J    
           ii2 = 1;
           for ii = i:I
              ndx = ii-1+mm*(jj-1);
              if ~sourceRegion(ndx+1) flag = 1; break; end
              ndx2=ii2-1+m*(jj2-1);
              if ~toFill(ndx2+1)
                  err=img(ndx+1) - Ip(ndx2+1); patchErr = patchErr+err*err; ndx=ndx+mmnn; ndx2=ndx2+mn;
                  err=img(ndx+1) - Ip(ndx2+1); patchErr = patchErr+err*err; ndx=ndx+mmnn; ndx2=ndx2+mn;
  	              err=img(ndx+1) - Ip(ndx2+1); patchErr = patchErr+err*err;
              end    
              ii2 = ii2+1;
           end
           if flag == 1 break; end
           jj2 = jj2+1;
       end        
       % Reset
       if flag == 1 patchErr = 0.0; flag = 0; continue; end
       
       % Update
       if patchErr < bestErr
           bestErr = patchErr; 
	       best(1) = i; best(2) = I;
	       best(3) = j; best(4) = J;
       end
   end       
end
end


