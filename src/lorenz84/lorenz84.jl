export lorenz84!


function lorenz84!(du, u, p, t)
    a = 0.25
    F = 8.0
    b = 4.0
    G = 1.0

    du[1] = -u[2]^2 - u[3]^2 -a*u[1] + a*F
    du[2] = u[1]*u[2] - b*u[1]*u[3] - u[2] + G
    du[3] = b*u[1]*u[2] + u[1]*u[3] - u[3]
end