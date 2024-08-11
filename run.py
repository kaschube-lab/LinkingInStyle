import argparse

import torch

from src import create_paired_data, train_linking_nw, run_systematic_tuning, \
                extract_features, create_data_to_label, train_seg_model, \
                comp_vis_motion, gen_counterfactual

if __name__ == '__main__':
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    parser = argparse.ArgumentParser()

    # TODO: add help messages

    # create subparsers so that the argument depend on the task
    subparsers = parser.add_subparsers(dest='task')
    subparsers.required = True
    parser_create_paired_data = subparsers.add_parser('create_paired_data')
    parser_train_linking_nw = subparsers.add_parser('train_linking_nw')
    parser_run_systematic_tuning = subparsers.add_parser('run_systematic_tuning')
    parser_extract_features = subparsers.add_parser('extract_features')
    parser_create_data_to_label = subparsers.add_parser('create_data_to_label')
    parser_train_seg_model = subparsers.add_parser('train_seg_model')
    parser_comp_vis_motion = subparsers.add_parser('comp_vis_motion')
    parser_gen_counterfactual = subparsers.add_parser('gen_counterfactual')

    # create_paired_data
    parser_create_paired_data.add_argument('--dataset_name', type=str)
    parser_create_paired_data.add_argument('--classifier_name', type=str, default='rnet50')
    parser_create_paired_data.add_argument('--n_samples', type=int)
    parser_create_paired_data.add_argument('--truncation', type=float, default=0.7)
    parser_create_paired_data.add_argument('--gen_seed', type=int, default=0)
    parser_create_paired_data.add_argument('--partition', type=str, default='train')

    # train_linking_nw
    parser_train_linking_nw.add_argument('--dataset_name', type=str)
    parser_train_linking_nw.add_argument('--classifier_name', type=str, default='rnet50')
    parser_train_linking_nw.add_argument('--model_type', type=str, default='LR')
    parser_train_linking_nw.add_argument('--n_samples', type=int)
    parser_train_linking_nw.add_argument('--truncation', type=float, default=0.7)
    parser_train_linking_nw.add_argument('--gen_seed', type=int, default=0)

    # run_systematic_tuning
    parser_run_systematic_tuning.add_argument('--dataset_name', type=str)
    parser_run_systematic_tuning.add_argument('--classifier_name', type=str, default='rnet50')
    parser_run_systematic_tuning.add_argument('--n_steps', type=int, default=3)
    parser_run_systematic_tuning.add_argument('--samples_per_class', type=int)
    parser_run_systematic_tuning.add_argument('--outdir', type=str)
    parser_run_systematic_tuning.add_argument('--units', type=int, nargs='+', default=[0, 2])
    parser_run_systematic_tuning.add_argument('--model_type', type=str, default='LR')

    # extract_features
    parser_extract_features.add_argument('--dataset_name', type=str)
    parser_extract_features.add_argument('--img_size', type=int, default=256)
    parser_extract_features.add_argument('--w_path', type=str)
    parser_extract_features.add_argument('--partition', type=str, default='train')

    # create_data_to_label
    parser_create_data_to_label.add_argument('--dataset_name', type=str)
    parser_create_data_to_label.add_argument('--gen_seeds', type=int, nargs='+', default=[0, 100])
    parser_create_data_to_label.add_argument('--samples_per_class', type=int)
    parser_create_data_to_label.add_argument('--truncation', type=float, default=0.7)
    parser_create_data_to_label.add_argument('--partition', type=str, default='train')
    
    # train_seg_model
    parser_train_seg_model.add_argument('--dataset_name', type=str)
    parser_train_seg_model.add_argument('--n_epochs', type=int, default=2)
    parser_train_seg_model.add_argument('--img_size', type=int, default=128)
    parser_train_seg_model.add_argument('--model_size', type=int, default='S')

    # comp_vis_motion
    parser_comp_vis_motion.add_argument('--src_img_path', type=str)
    parser_comp_vis_motion.add_argument('--dst_img_path', type=str)
    parser_comp_vis_motion.add_argument('--sparse_corres_path', type=str)
    parser_comp_vis_motion.add_argument('--percentile', type=int, default=80)
    parser_comp_vis_motion.add_argument('--neigh', type=int, default=3)
    parser_comp_vis_motion.add_argument('--step_sparsify', type=int, default=5)

    # gen_counterfactual
    parser_gen_counterfactual.add_argument('--dataset_name', type=str)
    parser_gen_counterfactual.add_argument('--classifier_name', type=str, default='rnet50')
    parser_gen_counterfactual.add_argument('--model_type', type=str, default='LR')
    parser_gen_counterfactual.add_argument('--origin_class', type=int)
    parser_gen_counterfactual.add_argument('--target_class', type=int)
    parser_gen_counterfactual.add_argument('--manual_seed', type=int, default=0)
    parser_gen_counterfactual.add_argument('--out_dir', type=str)
    parser_gen_counterfactual.add_argument('--orig_img_path', type=str)
    parser_gen_counterfactual.add_argument('--orig_w_path', type=str)
    parser_gen_counterfactual.add_argument('--learning_rate', type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    if args.task == 'create_paired_data':
        create_paired_data(dataset_name=args.dataset_name,
                            classifier_name=args.classifier_name,
                            partition=args.partition,
                            n_samples=args.n_samples,
                            gen_seed=args.gen_seed,
                            truncation=args.truncation,
                            device=device)
    elif args.task == 'train_linking_nw':
        train_linking_nw(dataset_name=args.dataset_name,
                            classifier_name=args.classifier_name,
                            model_type=args.model_type,
                            n_samples=args.n_samples,
                            gen_seed=args.gen_seed,
                            truncation=args.truncation)
    elif args.task == 'run_systematic_tuning':
        run_systematic_tuning(classifier_name=args.classifier_name,
                              dataset_name=args.dataset_name,
                              n_steps=args.n_steps,
                              samples_per_class=args.samples_per_class,
                              outdir=args.outdir,
                              units=args.units,
                              model_type=args.model_type,
                              device=device)
    elif args.task == 'extract_features':
        extract_features(dataset_name=args.dataset_name,
                         w_path=args.w_path,
                         partition=args.partition,
                         img_size=args.img_size,
                         device=device)
    elif args.task == 'create_data_to_label':
        create_data_to_label(dataset_name=args.dataset_name,
                             gen_seeds=args.gen_seeds,
                             samples_per_class=args.samples_per_class,
                             truncation=args.truncation,
                             partition=args.partition,
                             device=device)
    elif args.task == 'train_seg_model':
        train_seg_model(dataset_name=args.dataset_name,
                        n_epochs=args.n_epochs,
                        img_size=args.img_size,
                        model_size=args.model_size,
                        device=device)
    elif args.task == 'comp_vis_motion':
        comp_vis_motion(src_img_path=args.src_img_path,
                        dst_img_path=args.dst_img_path,
                        sparse_corres_path=args.sparse_corres_path,
                        percentile=90,
                        neigh=3,
                        step_sparsify=5,
                        device=device)
    elif args.task == 'gen_counterfactual':
        gen_counterfactual(dataset_name=args.dataset_name,
                            classifier_name=args.classifier_name,
                            orig_img_path=args.orig_img_path,
                            orig_w_path=args.orig_w_path,
                            origin_class=args.origin_class,
                            target_class=args.target_class,
                            manual_seed=args.manual_seed,
                            out_dir=args.out_dir,
                            lr=args.learning_rate,
                            model_type=args.model_type,
                            device=device)
    else:
        raise ValueError(f'Unknown task {args.task}')
    
