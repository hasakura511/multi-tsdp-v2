from django.contrib import admin

from quotes.models import Group, Instrument, QuoteDaily


@admin.register(Group)
class GroupAdmin(admin.ModelAdmin):
    list_display = ['name', 'name_plural', ]


@admin.register(Instrument)
class InstrumentAdmin(admin.ModelAdmin):
    list_display = ['name', 'symbol', 'group', 'subscribed', ]
    list_filter = ['group', 'subscribed', ]


@admin.register(QuoteDaily)
class QuoteDailyAdmin(admin.ModelAdmin):
    list_display = ['instrument', 'date', 'open', 'high', 'low', 'close', 'volume', ]
    list_filter = ['instrument', ]
    date_hierarchy = 'date'
