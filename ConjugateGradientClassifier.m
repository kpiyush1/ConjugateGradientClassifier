BeginPackage["ConjugateGradientClassifier`"]


(* Standardizing the Data *)

featureNormalize[data_]:= Module[
	{x, xnorm, mu, sigma, m, n},
	x = N@data[[All,1;;-2]];
	xnorm = x;
	{m,n} = Dimensions[x];
	mu = Mean[x];
	sigma = StandardDeviation[x];
	Do[
		xnorm[[All,i]] = (x[[All,i]]-mu[[i]])/sigma[[i]];
		,
		{i,1,n}
	];
	xnorm
];

(* Choosing the hypothesis -- for Linear or Logistic Regression *)

Options[chooseHypothesis]={
	"HypothesisMethod" -> Automatic
};

chooseHypothesis[x_, theta_, opts:OptionsPattern[]]:= Module[
	{method},
	method = OptionValue["HypothesisMethod"];
	If[method === Automatic, method = "Basic"];
	ichooseHypothesis[method, x, theta]
];

ichooseHypothesis["Basic", x_, theta_]:= Module[
	{hypo},
	hypo = x.theta
]; 

ichooseHypothesis["Sigmoid", x_, theta_]:= Module[
	{hypo},
	hypo = 1./(1+Exp[-(x.theta)])
]; 

(* Computing Cost for linear and logistic regression for training data *)

Options[computeCost]= Options[icomputeCost] = {
	Method -> Automatic, 
	"HypothesisMethod" -> Automatic
};

computeCost[x_, theta_, lambda_, chooseHypothesis_, y_, m_, opts:OptionsPattern[]]:= Module[
	{method, hypomethod},
	method = OptionValue[Method];
	If[method === Automatic, method = "LeastSquares"];
	icomputeCost[method, x, theta, lambda, chooseHypothesis, y, m, FilterRules[{opts}, Options[icomputeCost]]]
];

icomputeCost["LeastSquares", x_, theta_, lambda_, chooseHypothesis_, y_, m_, opts:OptionsPattern[]]:= Module[
	{hypo, sqErrors, J0},
	hypo = chooseHypothesis[x, theta, FilterRules[{opts}, Options[chooseHypothesis]]];
	sqErrors = (Norm[hypo-y])^2;
	J0 = (1./(2m))* sqErrors + (lambda[[1,1]]*(1-lambda[[1,2]])/(2m))*((Total[theta^2]-(First[theta])^2)[[1]])
			+ (lambda[[1,2]]*lambda[[1,1]]/(2m)) * ((Total[theta]-(First[theta]))[[1]])
]; 

icomputeCost["Logistic", x_, theta_, lambda_, chooseHypothesis_, y_, m_, opts:OptionsPattern[]]:= Module[
	{hypo, J0},
	hypo = chooseHypothesis[x, theta, FilterRules[{opts}, Options[chooseHypothesis]]];
	J0 = (1./m)* (Total[-y*Log[hypo]-(1-y)*Log[1-hypo]])[[1]] + (lambda[[1,1]]*(1-lambda[[1,2]])/(2m))*((Total[theta^2]-(First[theta])^2)[[1]])
			+ (lambda[[1,2]]*lambda[[1,1]]/(2m)) * ((Total[theta]-(First[theta]))[[1]])
]; 

(* The Cost function and its Gradient *)

Options[costfunc]= Options[gradient] = {
	Method -> Automatic,
	"HypothesisMethod" -> Automatic
};

costfunc[data_, theta_, lambda_, opts:OptionsPattern[]]:= Module[
	{sqErrors, x, y, m, n, J},
	x = featureNormalize[data];
	x = Map[Prepend[#,1.]&, x];
	y = data[[All,{-1}]];
	{m, n} = Dimensions[x];
	J = computeCost[x, theta, lambda, chooseHypothesis, y, m,  FilterRules[{opts}, Options[computeCost]]]
];

gradient[data_, theta_, lambda_, opts:OptionsPattern[]] := Module[
	{hypothesis, x, y, m, n, thbasis, grad},
	x = featureNormalize[data];
	x = Map[Prepend[#,1]&, x];
	y = data[[All,-1]];
	{m, n} = Dimensions[x];
	hypothesis = chooseHypothesis[x, theta,  FilterRules[{opts}, Options[chooseHypothesis]]];
	thbasis =  ReplacePart[Table[1,{i,1,n},{j,1,1}],{1}->{0}];
	(* Both L1 and L2 regularizers included. L1 regularizer may cause problems as it is non-differentiable wrt \[Theta] for \[Theta] = 0. *)
	grad = (1./m)*Transpose[x].(hypothesis - y)+ (lambda[[1,1]]*(1-lambda[[1,2]])/m)* (theta*thbasis) + 
		(lambda[[1,2]]*lambda[[1,2]]/(2m))*Sign[theta]*thbasis
];

(* Main function for learning parameters for classification and regression *)

Options[ConjugateGradientClassifier] = {
	MaxIterations -> Automatic,
	Method -> Automatic,
	"HypothesisMethod" -> Automatic,
	"ConjugateGradientMethod"-> Automatic, 
	"GradientDescentThreshold" -> 0.1, (*as in Nocedal & Wright *)
	"MinimalGradientNorm" -> 10.^-6, (* 10^-5 used in Nocedal & Wright. Maybe refine later *)
	"MonitorStep" -> None	
};

ConjugateGradientClassifier[data_, costfunc_, gradient_, theta0_, lambda_, opts:OptionsPattern[]] := Module[
	{cost, costnew, searchdirec, theta, thetanew, grad, gradnew, mingradnorm, overlap, alpha, beta, monitor, 
		monitorstep, costhistory, maxiter, nu, delta, m, n, method, hypomethod, conjgradmethod},
	(*{m, n} = Dimensions[data];*)
	theta = theta0;
	mingradnorm = OptionValue["MinimalGradientNorm"];
	monitorstep = OptionValue["MonitorStep"];
	nu = OptionValue["GradientDescentThreshold"];
	monitor = (monitorstep =!= None);
	method = OptionValue[Method];
	hypomethod = OptionValue["HypothesisMethod"];
	If[method === Automatic, method = "LeastSquares"];
	If[hypomethod === Automatic, hypomethod = "Basic"];
	conjgradmethod = OptionValue["ConjugateGradientMethod"];
	cost = costfunc[data, theta, lambda,  FilterRules[{opts}, Options[costfunc]]];
	grad = gradient[data, theta, lambda,  FilterRules[{opts}, Options[gradient]]];
	If[monitor, costhistory = {cost}];
	maxiter = OptionValue[MaxIterations];
	If[maxiter === Automatic, maxiter = 100];
	searchdirec = -grad;(* Choose initial direction as -gradient *)
	(*Find the optimal step size by doing a line search *)
	Do[
		(* check if the squared residual is less than some specified value *)
		If[Norm[grad] < (mingradnorm*(1+Abs[cost])), Break[]];
		alpha = computeAlpha[data, theta, costfunc, gradient, lambda, searchdirec, cost, grad,  FilterRules[{opts}, Options[computeAlpha]]];
		thetanew = theta + alpha*searchdirec;
		costnew = costfunc[data, thetanew, lambda,  FilterRules[{opts}, Options[costfunc]]];
		delta = ((cost-costnew)/costnew);
		If[Abs[delta] < 10^-6, Break[]];
		gradnew = gradient[data, thetanew, lambda,  FilterRules[{opts}, Options[gradient]]];
		overlap = Abs[Transpose[gradnew].grad][[1,1]];
		overlap = overlap/Norm[grad];
		(*Compute \[Beta] by one of the many methods. Implement "restarts", i.e. make beta= 0 if successive gradients 
		not approximately orthogonal *)
		beta = If[overlap >= nu, 0, computeBeta[searchdirec, grad, gradnew,  FilterRules[{opts}, Options[computeBeta]]]]; 	
		theta = thetanew;
		grad = gradnew;
		cost = costnew;
		searchdirec = -grad + beta*searchdirec;
		,
		{iter, 1, maxiter}
	];
	cost = costfunc[data, theta, lambda,  FilterRules[{opts}, Options[costfunc]]];
	If[monitor, 
		costhistory = Transpose[{monitorstep*Range[0, Length[costhistory] - 1], costhistory}];
		Print[
			If[Length[costhistory] > 100,
				ListLogLinearPlot[Rest @ costhistory, Joined -> True]
				,
				ListPlot[costhistory, Joined -> True, PlotMarkers -> Automatic]
			];
		];
	];
	<|"MinCost[trainingdata]" -> cost, "theta_trained =" -> theta|>
];

(* Line-Search. We will mostly focus on a line search algorithm as explained in Nocedal and Wright that satisfies the strong Wolfe conditions, and 
consists of two stages. The first stage begins with a trial estimate for the step length (alpha) and keep increasing it until either an acceptable alpha is found 
or when an interval is found which contains the desired alpha. In the latter case, a function called "zoom" is called which successively decreases the size of the 
interval until an acceptable alpha is found. *)

Options[computeAlpha] = {
	Method -> Automatic, 
	"HypothesisMethod" -> Automatic, 
	"AlphaMethod" -> Automatic
};

computeAlpha[data_, theta_, costfunc_, gradient_, lambda_, searchdirec_, phi0_, grad0_, opts:OptionsPattern[]] := Module[
	{method},
	method = OptionValue["AlphaMethod"];
	If[method === Automatic, method = "StrongWolfeLineSearch"];
	icomputeAlpha[method, data, theta, costfunc, gradient, lambda, searchdirec, phi0, grad0,  FilterRules[{opts}, Options[computeAlpha]]]
];

Options[icomputeAlpha] = {
	Method -> Automatic, 
	"HypothesisMethod" -> Automatic,
	"MaxNumSteps" -> Automatic,
    "Alphamax" -> 30., 
	"rho" -> 10.^-4, (*used in Nocedal & Wright*)
	"sigma" -> 0.1
};

icomputeAlpha["Manual", data_, theta_, costfunc_, gradient_, lambda_, searchdirec_, phi0_, grad0_, opts:OptionsPattern[]] := Module[
	{alphaval},
	alphaval = 0.1;
	alphaval
];

icomputeAlpha["StrongWolfeLineSearch", data_, theta_, costfunc_, gradient_, lambda_, searchdirec_, phi0_, grad0_, opts:OptionsPattern[]] := Module[
	{direc, alphaval, alpha0, alphamax, alphaF, phigrad0, phiF, phigradF, alphaS, phiS, phigradS, thetaS, c1, c2, i, imax, cond1, cond2, cond3, flag},
	alphamax = OptionValue["Alphamax"];
	c1 =  OptionValue["rho"];
	c2 =  OptionValue["sigma"];
	imax = OptionValue["MaxNumSteps"]; 
	direc = searchdirec;
	alpha0 = 0.;
	alphaF = alpha0;
	alphaS = 1.; (*To check if this can be refined later *)
	phiF = phi0;
	phigrad0 = (Transpose[grad0].searchdirec)[[1,1]];
	phigradF = phigrad0;
	If[imax=== Automatic, imax = 50];
	flag = Do[
			thetaS = theta + alphaS* direc;
			phiS = costfunc[data, thetaS, lambda,  FilterRules[{opts}, Options[costfunc]]];
			phigradS = (Transpose[gradient[data, thetaS, lambda,  FilterRules[{opts}, Options[gradient]]]].searchdirec)[[1,1]];
	(* checking the sufficient decrease condition (W1). If not satisfied, then apply "zoom"*)
			cond1 = (phiS > phi0 + c1* (alphaS*phigrad0)) || (i > 1 && (phiS >= phiF));
			cond2 = (Abs[phigradS] <= -c2* phigrad0);
			cond3 = (phigradS >= 0);
			If[cond1
				,
				alphaval = zoom[data, theta, costfunc, gradient, lambda, searchdirec, phi0, phigrad0, alphaF, alphaS, phiF, phigradF, phiS, phigradS, 
					 FilterRules[{opts}, Options[zoom]]
				];
				Return[alphaval];
			];
	(*checking the curvature condition (W2). If satisfied, return the current value of alphaS. *)
			If[cond2, alphaval = alphaS; Return[alphaval];];
	(*If gradient is positive, then apply zoom again*)
			If[phigradS >= 0, 
				alphaval = zoom[data, theta, costfunc, gradient, lambda, searchdirec, phi0, phigrad0, alphaS, alphaF, phiS, phigradS,
					phiF, phigradF,  FilterRules[{opts}, Options[zoom]]
				]; 
				Return[alphaval];
			];
			alphaF = alphaS;
			phiF = phiS;
			phigradF = phigradS;
			alphaS = alphaS + (alphamax-alphaS)*RandomReal[];
			,
			{i, 1, imax}
		];
	If[SameQ[flag,Null], alphaS, alphaval]
]; 

(* Each iteration of zoom finds an iterate alpha_X between alpha_lo and alpha_hi, and replaces one of them by alpha_X in such a way that the conditions 
(a-c) in page 61 of Nocedal-Wright continue to hold. We use interpolation methods for zoom. They work better than the golden section and Fibonacci methods 
for functions whose gradients can be computed easily*)

Options[zoom] = {
	Method -> Automatic, 
	"HypothesisMethod" -> Automatic,
	"MaxTrials" -> 10,
	"rho" -> 10.^-4, (*used in Nocedal & Wright*)
	"sigma" -> 0.4
};

zoom[data_, theta_, costfunc_, gradient_, lambda_, searchdirec_, phi0_, phigrad0_, alphalo_, alphahi_, philo_, phigradlo_, phihi_, phigradhi_, opts:OptionsPattern[]] := Module[
{direc, alphal, alphah, phil, phigradl, phih, phigradh, alphaX, phiX, phigradX, thetaX, alphastar, c1, c2, jmax},
	c1 =  OptionValue["rho"];
	c2 =  OptionValue["sigma"];
	jmax=OptionValue["MaxTrials"];
	direc = searchdirec;
	alphal = alphalo;
	alphah = alphahi;
	phil = philo;
	phigradl = phigradlo;
	phih = phihi;
	phigradh = phigradhi;
	If[jmax=== Automatic, jmax = 10]; 
	Do[
		alphaX = interpolation[alphal, alphah, phil, phigradl, phih, phigradh];
		thetaX = theta + alphaX * direc;
		phiX = costfunc[data, thetaX, lambda,  FilterRules[{opts}, Options[costfunc]]];
		If[((phiX > phi0 + c1*alphaX*phigrad0)||(phiX >= philo)),
			alphah = alphaX;
			,
			phigradX = (Transpose[gradient[data, thetaX, lambda,  FilterRules[{opts}, Options[gradient]]]].direc)[[1,1]]; 
			If[Abs[phigradX] <= -c2 * phigrad0,              
				alphastar = alphaX;
				Break[];
			];				
			If[phigradX*(alphah-alphal) >= 0,
				alphah = alphal;
			];
			alphal = alphaX;
		];
	,
	{j,1,jmax}];
	alphaX
];

Options[interpolation] = {
	Method -> Automatic
};

interpolation[alphal_, alphah_, phil_, phigradl_, phih_, phigradh_, opts:OptionsPattern[]] := Module[
	{method},
	method = OptionValue[Method];
	If[method === Automatic, method = "CubicInterpolation"];
	iinterpolation[method, alphal, alphah, phil, phigradl, phih, phigradh]
];

Options[iinterpolation] = {
	"minstep" -> 10.^-4
};


(* Cubic Interpolation when function and gradient values at two values of alpha - alphalo & alphahi, given by 
 phi(alphalo), phi'(alphalo), phi(alphahi) & phi'(alphahi), are known *)
iinterpolation["CubicInterpolation", alphal_, alphah_, phil_, phigradl_, phih_, phigradh_, opts:OptionsPattern[]] := Module[
	{d1, d2, alphatemp, alphaj, minsteps},
	minsteps = OptionValue["minstep"];
	d1 = phigradl + phigradh - 3((phil-phih)/(alphal-alphah));
	d2 = Sign[alphah-alphal]* (d1^2-(phigradl*phigradh))^(1/2);
	alphatemp = alphah - ((alphah-alphal)*(phigradh+d2-d1)/(phigradh-phigradl+2*d2));
	Which[Abs[(alphah-alphatemp)]< minsteps, 
			alphaj = (alphal+alphah)/2.;
			,
			Abs[(alphatemp-alphal)] < minsteps,
			alphaj = (alphal+alphah)/2.;
			,
			True,
			alphaj = alphatemp;
	];
	alphaj
];

(* Different Conjugate Gradient Methods available *)

(* We choose as our default the method Hager-Zhang-Plus. Descent conditions guaranteed provided (strong) 
	Wolfe conditions are satisifed. If one uses restarts, then convergence guaranteed. *)

Options[computeBeta] = {
	"ConjugateGradientMethod" -> Automatic
};

computeBeta[searchdirec_, grad_, gradnew_, opts:OptionsPattern[]] := Module[
	{method},
	method = OptionValue["ConjugateGradientMethod"];
	If[method === Automatic, method = "Hager-Zhang-Plus"];
	icomputeBeta[method, searchdirec, grad, gradnew]
];

Options[icomputeBeta] = {
	"eta0" -> 0.1
};

icomputeBeta["Hager-Zhang-Plus", searchdirec_, grad_, gradnew_, opts:OptionsPattern[]] := Module[
	{direc, absdirec, deltagrad, absgrad, deltagradsqnorm, coeff, betapref, betanum, betadenom, beta, eta, eta0},
	direc = searchdirec;
	absdirec = Norm[direc];
	absgrad = Norm[grad];
	eta0 = OptionValue["eta0"];
	eta = -1/(absdirec * Min[eta0,absgrad]);
	deltagrad = (gradnew - grad);
	deltagradsqnorm = Norm[deltagrad]^2;
	coeff = (deltagradsqnorm)/(Transpose[deltagrad].direc)[[1,1]];
	betapref = deltagrad-(2 coeff* direc);
	betanum = (Transpose[betapref].gradnew)[[1,1]]; 
	betadenom = (Transpose[deltagrad].direc)[[1,1]];
	beta = Max[(betanum/betadenom),eta]
];

(* For the following, the strong Wolfe conditions  AND the sufficient descent conditions guarantee convergence. So, in this case one 
   needs a variant of the strong-Wolfe line search to ensure sufficient descent in addition. This can be achieved via the More-Thuente line search algorithm. 
	To implement later *)
	
icomputeBeta["Polak-Ribiere-Plus", searchdirec_, grad_, gradnew_] := Module[
	{deltagrad, betanum, betadenom, beta},
	deltagrad = gradnew- grad;
	betanum = (Transpose[gradnew].deltagrad)[[1,1]];
	betadenom = Norm[grad]^2;
	beta = Max[(betanum/betadenom),0]
];

(*Original method used for minimizing quadratic functions. Can be useful for some non-linear functions. *)
icomputeBeta["Hestenes-Stiefel-Plus", searchdirec_, grad_, gradnew_] := Module[
	{direc, deltagrad, betanum, betadenom, beta},
	direc = searchdirec;
	deltagrad = gradnew- grad;
	betanum = (Transpose[gradnew].deltagrad)[[1,1]];
	betadenom = (Transpose[deltagrad].direc)[[1,1]];
	beta = Max[(betanum/betadenom),0]
];

(*Fletcher-Reeves method -- supposedly not very robust *)
icomputeBeta["Fletcher-Reeves", searchdirec_, grad_, gradnew_] := Module[
	{betanum, betadenom, beta},
	betanum = Norm[gradnew]^2;
	betadenom = Norm[grad]^2;
	beta = betanum/betadenom
];


EndPackage[]