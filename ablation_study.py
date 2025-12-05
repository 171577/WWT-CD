"""
Ablation Study Script for SWCD + DGMA2
Automatically runs multiple experiments with different configurations
"""

import os
import subprocess
import json
from datetime import datetime


class AblationStudy:
    """Manages ablation study experiments"""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = []
        self.log_file = f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def run_experiment(self, exp_name, config_overrides):
        """
        Run a single experiment
        
        Args:
            exp_name: Experiment name
            config_overrides: Dict of config parameters to override
        """
        print(f"\n{'='*80}")
        print(f"Running Experiment: {exp_name}")
        print(f"{'='*80}")
        print(f"Config overrides: {config_overrides}")
        
        # Build command
        cmd = [
            "python", "train_dgma2_twostage.py",
            "--name", exp_name
        ]
        
        # Add base config
        for key, value in self.base_config.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Add overrides
        for key, value in config_overrides.items():
            cmd.extend([f"--{key}", str(value)])
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run training
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results (simplified - you may need to extract from logs)
            exp_result = {
                'name': exp_name,
                'config': config_overrides,
                'status': 'completed' if result.returncode == 0 else 'failed',
                'stdout': result.stdout[-500:] if result.stdout else '',  # Last 500 chars
                'stderr': result.stderr[-500:] if result.stderr else ''
            }
            
            self.results.append(exp_result)
            
            # Save results
            with open(self.log_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print(f"Experiment {exp_name} completed!")
            
        except Exception as e:
            print(f"Experiment {exp_name} failed: {e}")
            self.results.append({
                'name': exp_name,
                'config': config_overrides,
                'status': 'error',
                'error': str(e)
            })
    
    def run_beta_ablation(self):
        """Ablation study on β parameter"""
        print("\n" + "="*80)
        print("ABLATION 1: β Parameter (Similarity Injection Strength)")
        print("="*80)
        
        beta_values = [0.0, 0.1, 0.3, 0.5, 1.0]
        
        for beta in beta_values:
            self.run_experiment(
                exp_name=f"ablation_beta_{beta:.1f}",
                config_overrides={
                    'beta_init': beta,
                    'dam_per_scale': 2,
                    'dw_mdfm': False
                }
            )
    
    def run_dam_ablation(self):
        """Ablation study on DAM layers"""
        print("\n" + "="*80)
        print("ABLATION 2: Number of DAM Layers")
        print("="*80)
        
        dam_layers = [0, 1, 2]
        
        for num_dam in dam_layers:
            self.run_experiment(
                exp_name=f"ablation_dam_{num_dam}",
                config_overrides={
                    'beta_init': 0.3,
                    'dam_per_scale': num_dam,
                    'dw_mdfm': False
                }
            )
    
    def run_lightweight_ablation(self):
        """Ablation study on lightweight variants"""
        print("\n" + "="*80)
        print("ABLATION 3: Lightweight Variants")
        print("="*80)
        
        variants = [
            ('standard', {'dw_mdfm': False, 'dam_per_scale': 2}),
            ('dw_conv', {'dw_mdfm': True, 'dam_per_scale': 2}),
            ('reduced_dam', {'dw_mdfm': False, 'dam_per_scale': 1}),
            ('full_lightweight', {'dw_mdfm': True, 'dam_per_scale': 1})
        ]
        
        for variant_name, config in variants:
            config['beta_init'] = 0.3
            self.run_experiment(
                exp_name=f"ablation_lightweight_{variant_name}",
                config_overrides=config
            )
    
    def run_component_ablation(self):
        """Ablation study on model components"""
        print("\n" + "="*80)
        print("ABLATION 4: Component Analysis")
        print("="*80)
        
        # Note: This requires modifying the model code to support these variants
        # For now, we use beta=0 to simulate "no similarity injection"
        
        variants = [
            ('full_model', {'beta_init': 0.3, 'dam_per_scale': 2}),
            ('no_sim_injection', {'beta_init': 0.0, 'dam_per_scale': 2}),
            ('mdfm_only', {'beta_init': 0.0, 'dam_per_scale': 0}),
        ]
        
        for variant_name, config in variants:
            self.run_experiment(
                exp_name=f"ablation_component_{variant_name}",
                config_overrides=config
            )
    
    def generate_report(self):
        """Generate ablation study report"""
        print("\n" + "="*80)
        print("ABLATION STUDY REPORT")
        print("="*80)
        
        print(f"\nTotal experiments: {len(self.results)}")
        completed = sum(1 for r in self.results if r['status'] == 'completed')
        failed = sum(1 for r in self.results if r['status'] == 'failed')
        
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        
        print(f"\nDetailed results saved to: {self.log_file}")
        
        # Print summary table
        print("\n" + "-"*80)
        print(f"{'Experiment':<40} {'Status':<12} {'Config'}")
        print("-"*80)
        
        for result in self.results:
            config_str = ', '.join(f"{k}={v}" for k, v in result['config'].items())
            print(f"{result['name']:<40} {result['status']:<12} {config_str}")
        
        print("-"*80)


def main():
    """Main ablation study runner"""
    
    # Base configuration (shared across all experiments)
    base_config = {
        'dataroot': './datasets/LEVIR-CD',
        'batch_size': 16,
        'gpu_ids': 0,
        'stage1_epochs': 30,
        'stage2_epochs': 170,
    }
    
    # Create ablation study manager
    study = AblationStudy(base_config)
    
    # Run ablation studies
    print("="*80)
    print("SWCD + DGMA2 ABLATION STUDY")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Uncomment the experiments you want to run
    
    # 1. β parameter ablation
    study.run_beta_ablation()
    
    # 2. DAM layers ablation
    # study.run_dam_ablation()
    
    # 3. Lightweight variants
    # study.run_lightweight_ablation()
    
    # 4. Component analysis
    # study.run_component_ablation()
    
    # Generate final report
    study.generate_report()
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
if __name__ == "__main__":
    main()
