import uuid

from django.db import models
from phonenumber_field.modelfields import PhoneNumberField

from apps.common.helpers import get_display_name_for_slug


class BaseField:
    """Base field for the application defined model fields."""

    pass


class AppPhoneNumberField(BaseField, PhoneNumberField):
    """Applications version of the PhoneNumberField. To define app's functions."""

    pass


class AppSingleFileField(BaseField, models.FileField):

    def __init__(self, *args, **kwargs):
        from apps.common.models import COMMON_CHAR_FIELD_MAX_LENGTH

        kwargs.setdefault("upload_to", "files/")
        kwargs.setdefault("max_length", COMMON_CHAR_FIELD_MAX_LENGTH)

        super().__init__(*args, **kwargs)

    def pre_save(self, model_instance, add):
        file_field = getattr(model_instance, self.attname)

        if file_field:
            extension = file_field.name.rsplit(".", 1)[-1] if "." in file_field.name else ""
            new_filename = f"{uuid.uuid4().hex}"

            if extension:
                new_filename += f".{extension}"

            file_field.name = new_filename

        return super().pre_save(model_instance, add)


class AppSingleChoiceField(BaseField, models.CharField):
    """Application field for storing a single value from predefined choices."""

    def __init__(self, *args, choices_config=None, **kwargs):
        from apps.common.models import COMMON_CHAR_FIELD_MAX_LENGTH

        if choices_config is not None:
            if "choices" in kwargs:
                raise ValueError("Use either 'choices_config' or 'choices', not both.")

            kwargs["choices"] = choices_config

        kwargs.setdefault("max_length", COMMON_CHAR_FIELD_MAX_LENGTH)

        super().__init__(*args, **kwargs)
