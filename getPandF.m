function [P, F1] = getPandF(cm)

P=0;
TPR=0;
F1=0;

 for i = 1:26
     red=cm(i,:);
     kolona=cm(:,i);

     TP=red(1,i);
     doleP=TP;
     doleTPR=TP;
     for i=1:26
         doleP=doleP+red(1,i);
         doleTPR=doleTPR+kolona(i,1);
     end
   Ptre = TP/doleP;
   TPRtre = TP/doleTPR;
   P=P+(TP/doleP);
   TPR=TP+(TP/doleTPR);
   F1 = F1 + (Ptre * TPRtre)/(Ptre + TPRtre);
 end
 
 P=P/26;
 TPR=TPR/26;
 F1 = F1/26;
end