                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                1), tf.arg_max(Y_, 1))
    acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))
    tf.summary.scalar("accuarcy", acc_op)

def run(epochs):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        dataset = Imagenet.Cifar()
        trX, trY, teX, teY = dataset.getdata()

        filetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = "Networkfile/" + "convAENN_moreAElessNN"
        print(path)
        #path = "Networkfile/convAENN" + filetime
        saver = NNutils.save(path, sess)
        writer, merged = NNutils.graph(path, sess)

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:(batch_size)]

        st_time = datetime.now()

        for i in range(epochs):
            print(i, st_time)
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                summary, _, loss_nn, loss_ae, learning_rate, step = sess.run([merged, trainop, cost_NN, cost_AE, lr, global_step],
                                                                          feed_dict={ X: trX[start:end], Y: trY[start:end],
                                                                                      dropout_conv : 0.8, dropout_fc : 0.5})
                if step % 50 == 0:
                    writer.add_summary(summary, step)
                    print(step, datetime.now(), loss_nn, loss_ae, learning_rate)


            loss_nn, loss_ae, accuracy = sess.run([cost_NN, cost_AE, acc_op],
                                                  feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                  dropout_conv : 1.0, dropout_fc : 1.0})
            print("test results : ", accuracy, loss_nn, loss_ae)
            saver.save(sess, path + "/model.ckpt", step)

            # im = im.astype('uint8')
            # im = Image.fromarray(im[0])
            # im.save('convAENN.jpg')

        end_time = datetime.now()
        print("걸린 시간 = ", end_time - st_time)

        # for i in range(epochs):
        #     for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        #         sess.run([trainop_b], feed_dict={X: trX[start:end], Y: trY[start:end],
        #                                          dropout_conv : 0.8, dropout_fc : 0.5})
        #
        #     image, test_loss, loss, accuracy = sess.run([X_, cost, cost_NN, acc_op],
        #                                                feed_dict={X: teX[test_indices], Y: teY[test_indices],
        #                                                           dropout_conv : 1.0, dropout_fc : 1.0})
        #     print("test results : ", accuracy, test_loss, loss)



        # test_loss, accuracy = sess.run([cost, acc_op], feed_dict={X: teX[test_indices], Y: teY[test_indices],
        #                                                           dropout_conv : 1.0, dropout_fc : 1.0})
        # print("test results : ", accuracy, test_loss)


run(800)