function [X,fval10_5] = list_pred(quality_file,similarity_file)
% This function computes the trade-off front for diversity and quality of ranked list.
% Need MATLAB Global Optimization Toolbox
% Syntax: 	list_pred('quality.txt','similarity.txt')
%      
%     Inputs:
%           quality.txt File containing all the quality ratings
%			similarity.txt File containing the similarity kernel
%
%     Outputs:
%           fval10_5 Fitness of each ranking for both objectives
%			X Rankings on trade-off front
%


	
	popsize=12; %GA population size
	ngen=20; %Number of generations 
	num_items=100; %maximum number of items to consider for diversity computation (large number increases computation time)

	[x10_5,f10_5,fval10_5,exitflag,output10_5,population10_5,score10_5] = optim_listdpp(popsize,ngen,0,num_items,quality_file,similarity_file);
	f(:,1)=(f10_5(:,1)-min(f10_5(:,1)))./(max(f10_5(:,1))-min(f10_5(:,1)));
	f(:,2)=(f10_5(:,2)-min(f10_5(:,2)))./(max(f10_5(:,2))-min(f10_5(:,2)));


	%Plot trade-off front
	figure;
	plot(f(:,1),f(:,2),'o-');    
    [~,idx]=min(f(:,1).^2+f(:,2).^2);
    hold on;plot(f(idx,1),f(idx,2),'*r');
	xlabel('Quality');
	ylabel('Diversity');
	set(gca,'FontWeight','bold','FontSize',14);
	
    %Display ranking for indifference curve
    x=f10_5(idx,3:end);
    [~,iX]=sort(x,'descend');
	disp(iX);
	X=[];
	b=(size(f10_5));
	for i=1:b(1)
		[~,X(i,:)]=sort(f10_5(i,3:end),'descend');
	end

end



function [x,f,fval,exitflag,output,population,score,options] = optim_listdpp(PopulationSize_Data,StallGenLimit_Data,TolFun_Data,num_items,outfile,similarityfile)
	
	%quality ratings
	outf=load(outfile);
	outrank=outf';	
	outrank=min(outrank)+(outrank-min(outrank))./(max(outrank)-min(outrank));
	num_doc=length(outrank);%number of documents
	simila=load(similarityfile);
	idx=1:num_doc;
	
	if(num_doc<num_items)
		num_items=num_doc;
	end
	

	%Initialize population
	x1=greedy(num_doc,outrank);
	x2=greedydpp(simila);
	pop= create_pop(num_doc,PopulationSize_Data,x1,x2);


	% Start with the default options
	options = gaoptimset();
	% Modify options setting
	options = gaoptimset(options,'PopulationSize', PopulationSize_Data);
	options = gaoptimset(options,'Generations', StallGenLimit_Data);
	options = gaoptimset(options,'TolFun', TolFun_Data);
	options = gaoptimset(options,'Display', 'iter');
	options = gaoptimset(options,'PlotFcns', {@gaplotpareto});
	options = gaoptimset(options,'OutputFcns', { [] });
	options = gaoptimset(options,'UseParallel', true);
	options= gaoptimset(options,'InitialPopulation',pop);

	tic
	[x,fval,exitflag,output,population,score] = ...
	gamultiobj(@(x)dpp_fitness(x,outrank,simila,num_items),num_doc,[],[],[],[],0,1,options);
	y(:,idx)=x;
	f=sortrows([fval y]);
	X=find(f(1,3:end)==1);
	toc
end


function pop= create_pop(num_doc,popsize,X1,X2)
%Creates random population and inserts three genes
	pop=rand(popsize,num_doc);
	pop(1,:)=X1;
	pop(2,:)=X2;
end


function x=greedy(num_doc,outrank)
%Returns greedy list sorted by applauds
	stepsize=1/num_doc;
	[~, idx]=sort(outrank);
	x(idx)=0:stepsize:1-stepsize;
end

function [setc, dpval]=rank_dpp(simil)
tic()
setc=[];
num_elem=length(simil);
%%% This function takes as input a similarity matrix and returns greedy dpp sorted list
	idx=1:num_elem;
	min2set=min(simil(:));
	[row, column] = find(simil == min2set,1);
	setc(1:2)=[row column];
	idx(setc)=[];
	denom=det(simil+eye(num_elem));
	denom=1;
	dpval(1:2)=[1/denom (1-min2set^2)/denom];
	for j=3:num_elem
		fff=[];
		for i=1:length(idx)
			vec=[setc idx(i)];
			temp=simil(vec,vec);
			fff(i)=det(temp)/denom;
		end
		[a b]=max(fff);
		dpval(j)=a;
		setc(j)=idx(b);
		idx(b)=[];
	end

	%orderset=setc(end:-1:1);
	plot(dpval,'o-');
toc()
end

function x=greedydpp(simila)

	num_doc=length(simila);
	stepsize=1/num_doc;
	[setc, ~]=rank_dpp(simila);
	id=setc;
	%x(id)=0:stepsize:1-stepsize;
	x(id)=1-stepsize:-stepsize:0;

end




function f= dpp_fitness(x,outrank,similarity,num_items)

	num_pred=length(x);
	[~, idx]=sort(x,'descend');

	f1=0;
	f2=0;

	f1=dcg((num_pred:-1:1)',outrank(idx));
	
	for i=1:num_items
		X=idx(1:i);
		temp=similarity(X,X);	
		f2=f2+log10(det(temp))/i;
		%plot(i,f2,'o');hold on;
	end

	f=[-f1 -f2];
end

function ndcg=dcg(ysc,ytar)
	%ysc is any criteria on which it should be sorted, higher is better
	%ytar is the relevance value between 0 to 1
	yt=[ysc ytar];
	v1=0;
	v2=0;
	[~,idx1]=sort(ysc,'descend');
	[~,idx2]=sort(ytar,'descend');

	yscw=yt(idx1,:);
	yideal=yt(idx2,:);

	for i=1:length(ysc)
		v1=v1+((2^yscw(i,2))-1)/(log2(i+1));
		v2=v2+((2^yideal(i,2))-1)/(log2(i+1));
	end
	ndcg=v1/v2;
end







