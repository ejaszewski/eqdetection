class SE_net(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv_ds1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=11, padding=5)
        self.conv_ds2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, padding=3)
        self.conv_ds3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv_ds4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.conv_us_p1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_us_p2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2)
        self.conv_us_p3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, padding=3)
        self.conv_us_p4 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=11, padding=5)
        
        self.conv_us_s1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_us_s2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2)
        self.conv_us_s3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, padding=3)
        self.conv_us_s4 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=11, padding=5)
        
        self.conv_us_c1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_us_c2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2)
        self.conv_us_c3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, padding=3)
        self.conv_us_c4 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=11, padding=5)
                
        self.mp1 = nn.MaxPool1d(2)
        
        self.us1 = nn.Upsample(scale_factor=2)
        
        self.cse_ds5 = ChannelSELayer(64, 16)
        self.cse_us_p5 = ChannelSELayer(16, 4)
        self.cse_us_s5 = ChannelSELayer(16, 4)
        self.cse_us_c5 = ChannelSELayer(16, 4)
            
    def forward(self, x):
        # Combined encoder
        x = self.conv_ds1(x)
        x = self.mp1(x)
        x = self.conv_ds2(x)
        x = self.mp1(x)
        x = self.conv_ds3(x)
        x = self.mp1(x)
        x = self.conv_ds4(x)
        x = self.mp1(x)
        
        # se network
        x = self.cse_ds5(x)
        
        # P-arrival Decoder
        p = self.us1(x)
        p = self.conv_us_p1(p)
        p = self.us1(p)
        p = self.conv_us_p2(p)
        
        # se network
        p = self.cse_us_p5(p)
        
        p = self.us1(p)
        p = self.conv_us_p3(p)
        p = self.us1(p)
        p = self.conv_us_p4(p)
        
        
        # S-arrival Decoder
        s = self.us1(x)
        s = self.conv_us_s1(s)
        s = self.us1(s)
        s = self.conv_us_s2(s)
        
        # se network
        s = self.cse_us_s5(s)
        
        s = self.us1(s)
        s = self.conv_us_s3(s)
        s = self.us1(s)
        s = self.conv_us_s4(s)
        
        # Coda Decoder
        c = self.us1(x)
        c = self.conv_us_c1(c)
        c = self.us1(c)
        c = self.conv_us_c2(c)
        
        # se network
        c = self.cse_us_c5(c)
        
        c = self.us1(c)
        c = self.conv_us_c3(c)
        c = self.us1(c)
        c = self.conv_us_c4(c)
        
        return p, s, c
