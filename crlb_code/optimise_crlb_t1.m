options=optimoptions('fmincon','OptimalityTolerance',1e-12,'StepTolerance',1e-20,'Display','none');%,'MaxFunctionEvaluations',inf,'MaxIterations',3000);
sigma=1; % can be set to any value, as it does not affect the optimum

%fix tr to a typical value
tr = 7.5;


% Calculate optimal ti for range of R1s
T1 = 0.2:0.2:2;
R1 = 1./T1;
%R1 = 0.5:0.5:3;
%bmaxes = 0.01:0.01:3;
%figure; hold all;
ti_opt = zeros(size(R1,2),2);
for i = 1:size(R1,2)
    %c = zeros(size(bmaxes));
    %for j = 1:size(bmaxes,2)
    %    c(j) = adc_crlb([1;adcs(i)],[0; bmaxes(j)],sigma);
    %end
    [ti_opt(i,:), val_opt] = fmincon(@(ti)t1invrec_crlb([1;R1(i)],ti,tr,sigma),[0.5; 1],[],[],[],[],[0 0],[7 7],[],options);
    %h = plot(bmaxes,sqrt(val_opt)./sqrt(c),'LineWidth',2); line_color = get(h,'Color');
    %h = scatter(max(b_opt),1,50,line_color,'filled'); set(h,'HandleVisibility','off');
end
%xlabel(['b (ms/μm^2)']); ylabel('SNR relative to optimum');
%legend(arrayfun(@(x) ['ADC = ' num2str(x) ' μm^2/ms'],adcs,'UniformOutput',false))




