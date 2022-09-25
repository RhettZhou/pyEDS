function res = psnr4d(hat, star, std)

    if nargin < 3
        std = std2(star);
    end

%     res = 10 * ...
%           log(std^2 / mean((hat(:) - star(:)).^2)) ...
%           / log(10);
    normalized = norm(hat(:) - star(:))^2/norm(star(:))^2;
    res = 10 * ...
          log(std^2 / normalized) ...
          / log(10);

end
