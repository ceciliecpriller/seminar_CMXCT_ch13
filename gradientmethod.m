function [] = gradientmethod(step_size_method, maxiterations)

%set default step size method
if isempty(step_size_method)
   step_size_method = 'exact line search';
end

%test example created using AIR Tools II,
%generation of a 2-D parallel-beam tomography problem
N = 200; theta = 0:5:179; p = 2*N;
[A, b, x] = paralleltomo(N,theta,p);
x = reshape(x,N,N);
subplot(2, 2, 1);
imshow(x, []);
title('Original Phantom');

%define starting point
x_0 = zeros(N*N, 1);

%define tolerance
epsilon = 1e-3;

%define provisory step size
t = 0.25;

%define current number of iterations
iteration = 0;

%define strating vector
currentX = x_0;
x_minus1 = zeros(N*N, 1);

%define objective function
f = @(x) (1/2) * (norm(A*x-b)^2);

grad_f = grad(currentX, A, b);

while and(norm(grad_f)>=epsilon, iteration<maxiterations)
    %define the correct step size depending on parameter 'step_size_method'
    switch step_size_method
        case 'exact line search'
            t = norm(grad(currentX, A, b))^2/norm(A*grad(currentX, A , b))^2;
        case 'backtracking line search'
            t = backtracking(currentX, f, A, b);
        case 'BB1 step sizes'
            if(iteration == 0)
               t = backtracking(currentX, f, A, b);
            else 
               t = bb(currentX, x_minus1, A, b, 1);
            end
        case 'BB2 step sizes'
            if(iteration == 0)
               t = backtracking(currentX,f, A ,b);
            else 
               t = bb(currentX, x_minus1, A, b, 2);
            end
        otherwise
    end
    
    %calculate this iteration's x
    newX = currentX- t*grad_f;

    if ~isfinite(newX)
        error('x is inf or NaN')
    end

    %update values
    iteration = iteration + 1;
    x_minus1 = currentX;
    currentX = newX;
    grad_f = grad(currentX, A, b);
end 

x_grad = reshape(currentX, [N, N]);
subplot(2, 2, 2);
imshow(x_grad, []);
title(sprintf('Approximated phantom in %d iterations', iteration));

%calculates gradient at x
function g = grad(x, A, b)
g = A'*(A*x-b);

%Backtracking line search with alpha = 1e-2 and beta = 0.5
function t_k = backtracking(x, func, A, b)
t_k = 1;
alpha = 1e-2;
beta = 0.5;
while func(x-t_k*grad(x, A, b)) > func(x)-alpha * t_k * norm(grad(x, A ,b))^2
    t_k = t_k * beta;
end

%BB step sizes
%calculates the BB1 or BB2 step size depending on parameter 'version' 
function t_k = bb(x, x_minus1, A, b, version)
delta_y = grad(x, A, b) - grad(x_minus1, A , b);
delta_s = x - x_minus1;
if(version == 1)
    alpha_k = (delta_s' * delta_y) / norm(delta_s)^2;
    t_k = 1 / alpha_k;
else 
    t_k = (delta_s' * delta_y) / norm(delta_y)^2;
end





