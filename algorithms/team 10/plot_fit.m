function plot_fit(T,col)

global spike_train X spks

figure, hold on
plot(X(T,col))
plot(spike_train(T,col)+4)
plot(spks(T,col)+8)
xlim([0 range(T)])
axis off
legend('Calcium trace','Acutal Spikes','Predicted Spikes')
xlim([0 range(T)])
ylim([-3 20])
legend('boxoff')

