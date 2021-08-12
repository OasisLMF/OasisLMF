from typing import Dict

from .errors import HeaderTypeDescriptorError


class HeaderTypeDescriptor:
    """
    This class is responsible for defining the stream type in bytes.
    """
    def __get__(self, instance, owner) -> bytes:
        """
        Gets the type in bytes based off the instance.stream_type.

        Args:
            instance: (Any) the class instance that is currently calling the descriptor
            owner: (Any) the class that the instance belongs to

        Returns: (bytes) the stream type of the instance in bytes
        """
        if instance.stream_type == 1:
            return b'\x01\x00\x00\x00'
        elif instance.stream_type == 2:
            return b'\x02\x00\x00\x00'
        raise HeaderTypeDescriptorError(message=f"{instance.stream_type} type is not supported for header type")

    def __set__(self, instance, value) -> None:
        """
        Should set the descriptor but will raise error as this is not permitted in this descriptor.

        Args:
            instance: (Any) the class instance that is currently calling the descriptor
            value: (Any) the class that the instance belongs to

        Returns: None
        """
        raise HeaderTypeDescriptorError(message="")


class FileMapDescriptor:
    """
    This class is responsible for defining the file type based off the extension.
    """
    def __get__(self, instance, owner) -> Dict[str, str]:
        """
        Gets the file map based of of the extension of the instance attribute.

        Args:
            instance: (Any) the class instance that is currently calling the descriptor
            owner: (Any) the class that the instance belongs to

        Returns: (Dict[str, str]) the file map mapping the file name to the data type
        """
        return {
            "vulnerabilities": f"/static/vulnerability.{instance.extension.value}",
            "footprint": f"/static/footprint.{instance.extension.value}",
            "damage_bin": f"/static/damage_bin_dict.{instance.extension.value}",
            "events": f"/input/events.{instance.extension.value}",
            "items": f"/input/items.{instance.extension.value}"
        }

