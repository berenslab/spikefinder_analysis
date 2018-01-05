function plot_pred(T,col)

global spike_train X spks

figure, hold on
plot(X(T,col))
plot(spks(T,col)+4)
xlim([0 range(T)])
axis off
legend('Calcium trace','Predicted Spikes')
xlim([0 range(T)])
ylim([-3 14])
legend('boxoff')

