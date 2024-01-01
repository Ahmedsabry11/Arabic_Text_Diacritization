    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = ToTensor()

    def __len__(self):
        return len(self.data)
