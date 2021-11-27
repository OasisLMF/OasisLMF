from contextlib import ExitStack
from typing import Any


class FootprintReadDescriptor:
    """
    This class is responsible for checking that the class tethered to this descriptor has all the correct functions
    and attributes. It then runs the read function and checks that the correct attributes have been attributed.
    """
    INTRO: str = "You've coded a custom footprint read function in your class and attached the FootprintAdapterMixin."

    @staticmethod
    def _check_output_data(instance: Any) -> None:
        """
        Checks to ensure that the tethered object has the correct attributes after running the read function.

        Args:
            instance: (Any) the class tethered to this descriptor

        Returns: None
        """
        footprint_index_check: str = "present"
        footprint_check: str = "present"
        num_intensity_bins_check: str = "present"

        if hasattr(instance, 'num_intensity_bins') is False:
            num_intensity_bins_check = "missing"
        if hasattr(instance, 'footprint') is False:
            footprint_check = "missing"
        if hasattr(instance, 'footprint_index') is False:
            footprint_index_check = "missing"

        if "missing" in [footprint_check, footprint_index_check, num_intensity_bins_check]:
            raise NotImplementedError(
                f"""
                \n{FootprintReadDescriptor.INTRO}
                However, this read function does not result in creating all of the needed class attributes please check
                below which attributes the custom read function hasn't created\n
                num_intensity_bins: {num_intensity_bins_check}\n
                footprint: {footprint_check}\n
                footprint_index: {footprint_index_check}\n
                """
            )

    @staticmethod
    def _check_input_attributes(instance: Any) -> None:
        """
        Checks to ensure that the tethered object has the correct attributes and functions before running the read
        function.

        Args:
            instance: (Any) the class tethered to this descriptor

        Returns: None
        """
        if hasattr(instance, "static_path") is False:
            raise AttributeError(
                f"""
                \n{FootprintReadDescriptor.INTRO}
                However, your class does not have an attribute called "static_path". Please add this attribute which is 
                a string pointing to the path where the footprint data is. 
                """
            )
        read_function = getattr(instance, "read", None)
        if read_function is None:
            raise AttributeError(
                f"""
                \n{FootprintReadDescriptor.INTRO}
                However, your class does not have a function called "read". Please add this function and ensure that 
                this function populates the attributes for your class below:
                \nnum_intensity_bins
                \nfootprint
                \nfootprint_index\n
                """
            )
        if not callable(read_function):
            raise AttributeError(
                f"""
                \n{FootprintReadDescriptor.INTRO}
                However, you have added an attribute called "read". But this has to be function. This function also 
                has to populate the attributes for your class below:
                num_intensity_bins
                footprint
                footprint_index
                """
            )
        get_event_function = getattr(instance, "get_event", None)
        if get_event_function is None:
            raise AttributeError(
                f"""
                \n{FootprintReadDescriptor.INTRO}
                However, your class does not have a function called "get_event". Please add this function and ensure 
                that the function returns a specific event based off the input parameter "event_id".
                """
            )
        if not callable(get_event_function):
            raise AttributeError(
                f"""
                \n{FootprintReadDescriptor.INTRO}
                However, you have added an attribute called "get_event". Please add this as a function and ensure 
                that the function returns a specific event based off the input parameter "event_id".
                """
            )

    @staticmethod
    def _add_stack(instance: Any) -> None:
        """
        Adds an ExitStack to the instance under the attribute "stack".

        Args:
            instance: (Any) the class tethered to this descriptor

        Returns: None
        """
        instance.stack = ExitStack()

    def __get__(self, instance, owner) -> None:
        """
        Fires the whole process when the descriptor is called.

        Args:
            instance: (Any) the class tethered to this descriptor
            owner: (Any) the class that the instance belongs to

        Returns: None
        """
        self._add_stack(instance=instance)
        self._check_input_attributes(instance=instance)
        instance.read()
        self._check_output_data(instance=instance)
