txtfile = 'pointcloud.txt';
A = importdata(txtfile);
dimensions = size(A);
numrows = dimensions(1);

filtered = A(any(A,2),:);
x = filtered(:,1);
y = filtered(:,2);
z = filtered(:,3);

scatter3(x,y,z);
