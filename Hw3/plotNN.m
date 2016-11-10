function [] = plotNN(name,data,class,I,1)
%Plots a 2D scatterplot with x1,x2 and the corresponding classes in color
%and shape.

group = [class (I-1)'==class];
cmap = hsv(10);
gscatter(data(:,1),data(:,2),group,'bbmm','*o*o')
legend('Class 0: wrong','Class 0: correct', 'Class 1: wrong', 'Class 1: correct','Location','northwest')
title(strcat('Dataset ',name))
xlabel('x_1') % x-axis label
ylabel('x_2') % y-axis label



end

