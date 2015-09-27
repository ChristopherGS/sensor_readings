# app/users/tests.py
from flask import url_for
from flask.ext.login import current_user

from app.test_base import BaseTestCase
from .models import User


class UserViewsTests(BaseTestCase):
    def test_users_can_login(self):
        User.create(name="Joe", email="joe@joes.com", password="12345")

        with self.client:
            response = self.client.post(url_for("users.login"),
                                        data={"email": "joe@joes.com",
                                              "password": "12345"})

            self.assert_redirects(response, url_for("sensors.index"))
            self.assertTrue(current_user.name == "Joe")
            # self.assertFalse(current_user.is_anonymous())


    def test_users_can_logout(self):
        User.create(name="Joe", email="joe@joes.com", password="12345")

        with self.client:
            self.client.post(url_for("users.login"),
                             data={"email": "joe@joes.com",
                                   "password": "12345"})
            self.client.get(url_for("users.logout"))

            # self.assertTrue(current_user.is_anonymous())