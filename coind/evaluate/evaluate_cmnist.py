

def main(cfg):
    
    ########## Hyperparameters and settings ##########
    set_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    logger = logging.getLogger(__name__)
    ########## Dataset ##########
    
    train_dataset = instantiate(cfg.dataset.train_dataset)
    val_dataset = instantiate(cfg.dataset.val_dataset)
   
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,**cfg.dataset.train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,**cfg.dataset.val_dataloader)

    ########## Load diffusion model #############

    model = instantiate(cfg.diffusion.model)
    noise_schedule = instantiate(cfg.diffusion.noise_scheduler)
    model.load_state_dict(torch.load(cfg.diffusion_checkpoint_path)["model"])
    sampler = CondDDIMPipeline(model, noise_scheduler)

    ########## Metrics #############
    cs_metric = CS()
    quality = Quality()
    jsd = JSD()

    for pbar in [ tqdm(val_dataloader, desc="Val"),tqdm(train_dataloader, desc="Train")]:
        
        for batch in pbar:
            #construct query for AND, NOT(1) and NOT(2) 
                # Generate the image
                # CS
                # Quality


            #evaluate JSD
            


            #logger

        #log the metrics similar to the ones seen in the paper
        

if __name__ == "__main__":
    main()