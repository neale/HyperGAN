def ae_code():

    print ("==> pretraining Autoencoder")
    for i in range(0):
        ae_losses, layers, samples = [], [], []
        # lets hardcode then optimize
        args.id = 0
        netG.zero_grad()
        netE.zero_grad()
        x = sample_x(args, param_gen[0], 0) # sample
        x_enc = netE(x)
        x_fake = ops.gen_layer(args, netG, x_enc)
        x_fake = x_fake.view(*args.shapes[0])
        ae_loss = F.mse_loss(x_fake, x)
        ae_loss.backward(retain_graph=True)
        ae_losses.append(ae_loss.cpu().data.numpy()[0])
        optimizerE.step()
        optimizerG.step()

        args.id += 1
        netG.zero_grad()
        netE.zero_grad()
        xl = x_fake
        xl = xl.view(-1, *args.shapes[1][1:])
        xl_target = sample_x(args, param_gen[1], 1)
        xl_enc = netE(xl)
        xl_fake = ops.gen_layer(args, netG, xl_enc)
        xl_fake = xl_fake.view(*args.shapes[1])
        ae_loss = F.mse_loss(xl_fake, xl_target)
        ae_loss.backward()
        ae_losses.append(ae_loss.cpu().data.numpy()[0])
        optimizerE.step()
        optimizerG.step()

        if i % 500 == 0:
            norm_x = np.linalg.norm(x.data)
            norm_z = np.linalg.norm(x_fake.data)

            norm_xl = np.linalg.norm(xl_target.data)
            norm_zl = np.linalg.norm(xl_fake.data)

            cov_x_z = cov(x, x_fake).data[0]
            cov_xl_zl = cov(xl_target, xl_fake).data[0]
            print (ae_losses, 'CONV-- G: ', norm_z, '-->', norm_x, 
                    'LINEAR-- G: ', norm_zl, '-->', norm_xl)
            """
            utils.plot_histogram([x.cpu().data.numpy().flatten(),
                                  x_fake.cpu().data.numpy().flatten()],
                                  save=False, id='conv iter {}'.format(i))
            utils.plot_histogram([xl_target.cpu().data.numpy().flatten(),
                                  xl_fake.cpu().data.numpy().flatten()],
                                  save=False, id='linear iter {}'.format(i))
            """
def gan_train_code():

    for iteration in range(0, args.epochs):
        start_time = time.time()

        """ Update AE """
        # print ("==> autoencoding layers")
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        print ('==> updating AE') 
        for batch_idx, (data, target) in enumerate(mnist_train):
            ae_losses = []
            netE.zero_grad()
            netG.zero_grad()
            args.id = 0  # reset
            x = sample_x(args, param_gen[0], 0)
            z1 = ops.gen_layer(args, netG, netE(x))
            z1 = z1.view(*args.shapes[0])
            z1_loss = F.mse_loss(z1, x)
            args.id = 1
            x2 = sample_x(args, param_gen[1], 1)
            z2 = z1.view(-1, *args.shapes[1][1:])
            z2 = ops.gen_layer(args, netG, netE(z2))
            z2 = z2.view(*args.shapes[1])
            z2_loss = F.mse_loss(z2, x2)
            correct, loss = train_clf(args, [z1, z2], data, target, val=True)
            scaled_loss = (loss*.05) + z2_loss + z1_loss
            scaled_loss.backward(retain_graph=True)
            optimizerE.step()
            optimizerG.step()
            ae_losses.append(z1_loss.cpu().data.numpy()[0])
            ae_losses.append(z2_loss.cpu().data.numpy()[0])
            clf_loss = loss.cpu().data.numpy()[0]
            acc = correct / (float(len(target)))

            # Update Adversary 
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # print ('==> updating D')
            layers, d_losses, w1_losses = [], [], []
            args.id = 0  # reset
            x = sample_x(args, param_gen[0], id=0)
            z1 = ops.gen_layer(args, netG, netE(x))
            z1 = z1.view(*args.shapes[0])
            d_real, d_fake, gp = train_adv(args, netD, x, z1)
            d_real.backward(torch.Tensor([-1]).cuda(), retain_graph=True)
            d_fake.backward(retain_graph=True)
            gp.backward()
            optimizerD.step()
            w1_losses.append((d_real - d_fake).cpu().data.numpy()[0])
            d_losses.append((d_fake - d_real + gp).cpu().data.numpy()[0])
            layers.append(z1)

            args.id = 1
            x2 = sample_x(args, param_gen[1], 1)
            z2 = z1.view(-1, *args.shapes[1][1:])
            z2 = ops.gen_layer(args, netG, netE(z2))
            z2 = z2.view(*args.shapes[1])
            d_real, d_fake, gp = train_adv(args, netD, x2, z2)
            d_real.backward(torch.Tensor([-1]).cuda(), retain_graph=True)
            d_fake.backward(retain_graph=True)
            gp.backward()
            optimizerD.step()
            w1_losses.append((d_real - d_fake).cpu().data.numpy()[0])
            d_losses.append((d_fake - d_real + gp).cpu().data.numpy()[0])
            layers.append(z2)

            # correct, loss = train_clf(args, layers, data, target, val=True)
            # loss.backward()
            # optimizerD.step()

            # print ("==> updating g")
            g_losses = []
            args.id = 0
            g_cost = train_gen(args, netG, netD)
            g_cost.backward(torch.Tensor([-1]).cuda())
            g_losses.append(g_cost.cpu().data.numpy()[0])
            optimizerG.step()
            args.id = 1
            g_cost = train_gen(args, netG, netD)
            g_cost.backward(torch.Tensor([-1]).cuda())
            g_losses.append(g_cost.cpu().data.numpy()[0])
            optimizerG.step()

            # Write logs
            if batch_idx % 100 == 0:
                print ('==> iter: ', iteration)
                print('AE cost', ae_losses)
                # save_dir = './plots/{}/{}'.format(args.dataset, args.model)
                # path = 'params/sampled/{}/{}'.format(args.dataset, args.model)
                # utils.save_model(args, netE, optimizerE)
                # utils.save_model(args, netG, optimizerG)
                # utils.save_model(args, netD, optimizerD)
                # print ("==> saved model instances")
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # samples = netG(z)
                print ("****************")
                print('Iter ', batch_idx, 'Beta ', args.beta)
                print('D cost', d_losses)
                print('G cost', g_losses)
                print('AE cost', ae_losses)
                print('W1 distance', w1_losses)
                print ('clf (acc)', acc)
                print ('clf (loss', clf_loss)
                # print ('filter 1: ', layers[0][0, 0, :, :], layers[1][:, 0])
                print ("****************")
