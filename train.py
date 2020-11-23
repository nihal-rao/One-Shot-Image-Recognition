import torch

from torch.utils.tensorboard import SummaryWriter
from data_loader import get_train_valid_loader, get_test_loader
from model import Siamese

def main():
    
    siamese_model = Siamese()
    data_dir = 'omniglot-py'

    batch_size = 4
    num_train = 30000
    augment= True
    way = 20
    trials = 300
    epochs = 10

    train_loader, val_loader = get_train_valid_loader(data_dir, batch_size, num_train, augment, way, trials, pin_memory=True)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(siamese_model.parameters(), lr=1e-3, momentum=0.9)

    writer = SummaryWriter()

    siamese_model.cuda()
    
    for i in range(epochs):
        siamese_model.train()
        #batch_count = 0
        avg_train_loss = 0.0
        for it, (img_1, img_2, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            img_1 = img_1.cuda()
            img_2 = img_2.cuda()
            labels = labels.cuda()
            preds = siamese_model(img_1, img_2)

            loss = criterion(preds, labels)
            
            avg_train_loss+=loss.item()
            writer.add_scalar('Loss_train', loss.item(), len(train_loader)*i + it)    
            
            loss.backward()
            optimizer.step()
            #batch_count+=1
            #print(batch_count)
        
        siamese_model.eval()
        count = 0
        with torch.no_grad():
            for ref_images,candidates  in val_loader:
                ref_images = ref_images.cuda()
                candidates = candidates.cuda()

                preds = siamese_model(ref_images, candidates)

                if torch.argmax(preds) == 0:
                    count+=1
        writer.add_scalar('Accuracy_validation', count/trials, i)

        print('Epoch {} | Val accuracy {}'.format(i, avg_train_loss/len(train_loader), count/trials))

    writer.flush()

if __name__ == '__main__':
    main()