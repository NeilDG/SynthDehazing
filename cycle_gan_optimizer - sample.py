class CycleGanOptimizer(nn.Module):
    def __init__(self, gan, identity_criterion, cycle_criterion, gan_criterion, 
        G_optimizer, G_args, D_optimizer, D_args, lambda_identity=5, lambda_cycle=10):
        super(CycleGanOptimizer, self).__init__()
        self.G_A = gan.G_A
        self.G_B = gan.G_B
        self.D_A = gan.D_A
        self.D_B = gan.D_B

        self.identity_criterion = identity_criterion(reduction='mean')
        self.cycle_criterion = cycle_criterion(reduction='mean')
        self.gan_criterion = gan_criterion(reduction='mean')

        self.G_optimizer = G_optimizer(
            itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
            **G_args
        )
        self.D_optimizer = D_optimizer(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()), 
            **D_args
        )

        self.lambda_identity = lambda_identity
        self.lambda_cycle = lambda_cycle

        self.register_buffer('labels_true', torch.Tensor())
        self.register_buffer('labels_false', torch.Tensor())

        self.A_image_pool = ImagePool()
        self.B_image_pool = ImagePool()

    def optimize(self, A, B):
        loss = {}
        
        fake_A = self.G_A(B)
        fake_B = self.G_B(A)

        # update G
        requires_grad(self.D_A, False)
        requires_grad(self.D_B, False)
        self.G_optimizer.zero_grad()

        G_A_identity_loss = self.identity_criterion(self.G_A(A), A) * self.lambda_identity
        G_B_identity_loss = self.identity_criterion(self.G_B(B), B) * self.lambda_identity

        G_A_cycle_loss = self.cycle_criterion(self.G_A(fake_B), A) * self.lambda_cycle
        G_B_cycle_loss = self.cycle_criterion(self.G_B(fake_A), B) * self.lambda_cycle

        is_A = self.D_A(fake_A)
        if self.labels_true.size() != is_A.size():
            self.labels_true = torch.ones_like(is_A)
            self.labels_false = torch.zeros_like(is_A)

        G_A_gan_loss = self.gan_criterion(is_A, self.labels_true)
        G_B_gan_loss = self.gan_criterion(self.D_B(fake_B), self.labels_true)

        G_loss = (G_A_identity_loss + G_B_identity_loss
            + G_A_cycle_loss + G_B_cycle_loss
            + G_A_gan_loss + G_B_gan_loss)

        G_loss.backward()
        self.G_optimizer.step()

        loss['G_A_identity_loss'] = G_A_identity_loss.item()
        loss['G_B_identity_loss'] = G_B_identity_loss.item()
        loss['G_A_cycle_loss'] = G_A_cycle_loss.item()
        loss['G_B_cycle_loss'] = G_B_cycle_loss.item()
        loss['G_A_gan_loss'] = G_A_gan_loss.item()
        loss['G_B_gan_loss'] = G_B_gan_loss.item()
        loss['G_loss'] = G_loss.item()

        # update D
        requires_grad(self.D_A, True)
        requires_grad(self.D_B, True)
        self.D_optimizer.zero_grad()

        fake_A = self.A_image_pool.query(fake_A)
        fake_B = self.B_image_pool.query(fake_B)

        D_A_real_gan_loss = self.gan_criterion(self.D_A(A), self.labels_true)
        D_A_fake_gan_loss = self.gan_criterion(self.D_A(fake_A.detach()), self.labels_false)
        D_B_real_gan_loss = self.gan_criterion(self.D_B(B), self.labels_true)
        D_B_fake_gan_loss = self.gan_criterion(self.D_B(fake_B.detach()), self.labels_false)
        
        D_A_loss = D_A_real_gan_loss + D_A_fake_gan_loss
        D_B_loss = D_B_real_gan_loss + D_B_fake_gan_loss

        D_loss = D_A_loss + D_B_loss

        D_loss.backward()
        self.D_optimizer.step()

        loss['D_A_real_gan_loss'] = D_A_real_gan_loss.item()
        loss['D_A_fake_gan_loss'] = D_A_fake_gan_loss.item()
        loss['D_B_real_gan_loss'] = D_B_real_gan_loss.item()
        loss['D_B_fake_gan_loss'] = D_B_fake_gan_loss.item()
        loss['D_A_loss'] = D_A_loss.item()
        loss['D_B_loss'] = D_B_loss.item()
        loss['D_loss'] = D_loss.item()

        return loss