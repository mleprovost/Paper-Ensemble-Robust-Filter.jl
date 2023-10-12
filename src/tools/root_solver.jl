export test_hybridsolver


# Combine Bisection and Newton method, this method has guaranteed convergence.
# Convergence order should be between linear and quadratic
# http://www.m-hikari.com/ams/ams-2017/ams-53-56-2017/p/hahmAMS53-56-2017.pdf
function test_hybridsolver(f, g, out, a, b; ϵx = 1e-7, ϵf = 1e-7, niter = 500)
    dxold = abs(b-a)
    dx = dxold
    fout = f(out)
    gout = g(out)


    fa = f(a)
    fb = f(b)

    @inbounds for j=1:niter
        # Bisect if Newton out of range, or not decreasing fast enough.
        if ((out - b)*gout - fout)*((out - a)*gout - fout) > 0.0 || abs(2.0*fout) >  abs(dxold * gout)
            dxold = dx
            dx = 0.5*(b-a)
            out = a + dx
            if isapprox(a, out, atol = ϵx)
                return out
            end
        else #Newton step is acceptable
            dxold = dx
            dx    = fout/gout
            tmp   = out
            out = out - dx
            if isapprox(tmp, out, atol = ϵx)
                return out
            end
        end
        # Convergence criterion
        if abs(dx) < ϵx || abs(fout) < ϵf
            return out
        end
        # The one new function evaluation per iteration
        fout = f(out)
        gout = g(out)
        # Maintain the bracket on the root
        if fout<0.0
            a = out
        else
            b = out
        end
    end

    print("Root solver did not converge after $(niter)")

    return out
end