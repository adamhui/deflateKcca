%%  eigVector Sort Normlization
function[eigVector,eigValue]=getSortedAndNormalizedEig(mat)
[eigVector,eigValue] = eig (mat);
eigValue = diag(real(eigValue));
[~, index] = sort(-abs(eigValue));
eigValue = eigValue(index);
eigVector = eigVector(:,index);
% eigVector=eigVector./sum(eigVector.*eigVector).^0.5;
end