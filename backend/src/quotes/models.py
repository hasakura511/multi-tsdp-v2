from django.db import models
from django.utils.translation import ugettext_lazy as _


class Group(models.Model):
    """Instrument group model."""

    name = models.CharField(_('name'), max_length=63)
    name_plural = models.CharField(_('name plural'), max_length=63, null=True, blank=True)

    class Meta:
        verbose_name = _('Group')
        verbose_name_plural = _('Groups')

    def __str__(self):
        return self.name


class Instrument(models.Model):
    """Instrument model."""

    name = models.CharField(_('name'), max_length=254, unique=True)
    symbol = models.CharField(_('symbol'), max_length=32, unique=True)
    exchange = models.CharField(_('exchange'), max_length=32)
    securure_type = models.CharField(_('secure type'), max_length=32)
    currency = models.CharField(_('currency'), max_length=32)
    group = models.ForeignKey(Group)
    decimal_places = models.IntegerField(_('decimal places'), default=5)
    subscribed = models.BooleanField(_('subscribed'), db_index=True, default=True)

    class Meta:
        verbose_name = _('Instrument')
        verbose_name_plural = _('Instruments')
        ordering = ('name', )

    def __str__(self):
        return self.name


class QuoteDaily(models.Model):
    """Quote daily model."""

    id = models.BigAutoField(primary_key=True)

    instrument = models.ForeignKey(Instrument)
    date = models.DateField(_('date'))

    open = models.DecimalField(_('open'), max_digits=10, decimal_places=5, default=0)
    high = models.DecimalField(_('high'), max_digits=10, decimal_places=5, default=0)
    low = models.DecimalField(_('low'), max_digits=10, decimal_places=5, default=0)
    close = models.DecimalField(_('close'), max_digits=10, decimal_places=5, default=0)
    volume = models.BigIntegerField(_('volume'), default=0)

    class Meta:
        verbose_name = _('Quote daily')
        verbose_name_plural = _('Quotes daily')

        unique_together = ('instrument', 'date')

    def __str__(self):
        return '{} - {}'.format(self.instrument, self.date)
