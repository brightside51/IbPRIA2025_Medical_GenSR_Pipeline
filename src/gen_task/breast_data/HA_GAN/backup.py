max_value_MSSIM = 0
min_value_MSSIM = 10000000000
sum_ssim = 0
for k in range(1):
    for i,dat in enumerate(train_loader):
        noise_1 = torch.randn((1, latent_dim)).cuda()
        fake_image_1 = G(noise_1,0)
        img1 = dat[0]

        msssim = msssim_3d(img1,fake_image_1)
        sum_ssim = sum_ssim+msssim
        
        if msssim > max_value_MSSIM:  #Save the 2 images that resulted the max MSSIM
            max_value_MSSIM = msssim
            samples_dir = "max_MSSIM_real"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
            
            samples_dir = "max_MSSIM_fake"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
        if msssim < min_value_MSSIM:         #Save the 2 images that resulted the min MSSIM
            min_value_MSSIM= msssim
            samples_dir = "min_MSSIM_real"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
            samples_dir = "min_MSSIM_fake"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
    print(f"MSSIM dataset + generated: {sum_ssim/((k+1)*(i+1))}")
    print(f"Min value for MSSIM dataset + generated: {min_value_MSSIM}")
    print(f"Max value for MSSIM dataset + generated: {max_value_MSSIM}")

N=10                                                                    #Fake
sum_ssim = 0
batch_size = 10  # Choose an appropriate batch size
max_value_MSSIM = 0
min_value_MSSIM = 10000000000

for i in range(0, N, batch_size):
    batch_ssim = 0
    for j in range(batch_size):
        noise_1 = torch.randn((1, latent_dim)).cuda()
        noise_2 = torch.randn((1, latent_dim)).cuda()
        fake_image_1 = G(noise_1,0)
        fake_image_2 = G(noise_2,0)

        if torch.cuda.is_available():
            img1 = fake_image_1.cuda()
            img2 = fake_image_2.cuda()
        
        score = msssim_3d(img1,img2)
        
        if score > max_value_MSSIM:  #Save the 2 images that resulted the max MSSIM
            max_value_MSSIM = score
            samples_dir = "max_MSSIM_img1"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
            
            samples_dir = "max_MSSIM_img2"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img2[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
        if score < min_value_MSSIM:         #Save the 2 images that resulted the min MSSIM
            min_value_MSSIM= score
            samples_dir = "min_MSSIM_img1"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
            samples_dir = "min_MSSIM_img2"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img2[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
        batch_ssim += score.item()  # Accumulate SSIM score for the batch

    sum_ssim += batch_ssim / batch_size  # Average SSIM score per batch

print(f"MSSIM Fake : {sum_ssim / (N / batch_size)}")
print(f"Min value for MSSIM Fake: {min_value_MSSIM}")
print(f"Max value for MSSIM Fake: {max_value_MSSIM}")

train_loader = torch.utils.data.DataLoader(trainset,batch_size = 2, shuffle=True, num_workers=workers)
