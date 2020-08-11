__all__ = [
    'RunExposure',
    'RunFmTest',
]


class RunExposure(ComputationStep):
    """
    - desc here -   
    """
    step_params = [ {'name': 'oed_location_csv', 'is_path': True, 'pre_exist': True,
                    'help': 'Source location CSV file path'},
                   ]

    '''
        parser.add_argument(
            '-s', '--src-dir', type=str, default=None, required=True,
            help='Source files directory - should contain the OED exposure files'
        )
        parser.add_argument(
            '-r', '--run-dir', type=str, default=None, required=False,
            help='Run directory - where files should be generated'
        )
        parser.add_argument(
            '-l', '--loss-factor', type=float, nargs='+',
            help='Loss factors to apply to TIVs - default is 1.0. Multiple factors can be specified.'
        )
        parser.add_argument(
            '-a', '--alloc-rule-il', type=int, default=KTOOLS_ALLOC_IL_DEFAULT,
            help='Ktools IL back allocation rule to apply - default is 2, i.e. prior level loss basis'
        )
        parser.add_argument(
            '-A', '--alloc-rule-ri', type=int, default=KTOOLS_ALLOC_RI_DEFAULT,
            help='Ktools RI back allocation rule to apply - default is 3, i.e. All level loss basis'
        )
        parser.add_argument(
            '-o', '--output-level', default='item',
            help='Level to output losses. Options are: item, loc, pol, acc or port.', type=str
        )
        parser.add_argument(
            '-f', '--output-file', default=None,
            help='Write the output to file.', type=str
        )

    '''


    '''
        def run_exposure(                                                                                                                                          
            self,
            src_dir,
            run_dir,
            loss_factors,
            net_ri,
            il_alloc_rule,
            ri_alloc_rule,
            output_level,
            output_file,
            include_loss_factor=True,
            print_summary=False):
    '''

    def run(self):
        return None



class RunFmTest(ComputationStep):
    """
    - desc here -   
    """
    step_params = [ {'name': 'oed_location_csv', 'is_path': True, 'pre_exist': True,
                    'help': 'Source location CSV file path'},
                   ]



    def run(self):
        return None
